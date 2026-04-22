"""Run the full test set through the running vLLM OpenAI-compatible server,
parse `<label>` tags, compute accuracy / precision / recall / F1, and dump both
a JSONL of per-row predictions and a Markdown report.

Assumes the vLLM server is already running at --url (default http://localhost:8000)
with a model served under --model-id. enable_thinking is forced to False via
chat_template_kwargs because our SFT format is `<reason>...</reason><label>0|1</label>`,
not Nemotron's default reasoning mode.

Concurrency is handled with a thread pool; vLLM batches requests internally so
pushing ~32 in flight keeps the GPU saturated.

Usage (inside host, server on port 8000):

    python -m src.evals.eval_via_vllm_server \
        --model-id sft-1875--max-model-len \
        --concurrency 32 \
        --out-md docs/sft_iter1875_eval.md \
        --out-jsonl results/eval/iter1875/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

LABEL_RE = re.compile(r"<label>\s*([01])\s*</label>")


def load_test(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            sys_msg = next((m for m in d["messages"] if m["role"] == "system"), None)
            ast_msg = next((m for m in d["messages"] if m["role"] == "assistant"), None)
            if sys_msg is None or ast_msg is None:
                continue
            m = LABEL_RE.search(ast_msg["content"])
            if not m:
                continue
            rows.append({
                "idx": i,
                "system": sys_msg["content"],
                "gold": int(m.group(1)),
                "gold_full": ast_msg["content"],
            })
    return rows


def parse_label(text: str) -> int | None:
    m = LABEL_RE.search(text)
    return int(m.group(1)) if m else None


def post_chat(url: str, model_id: str, sys_msg: str, max_tokens: int,
              timeout: float) -> dict:
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "system", "content": sys_msg}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def infer_one(url: str, model_id: str, row: dict, max_tokens: int,
              timeout: float, retries: int) -> dict:
    err = None
    for attempt in range(retries + 1):
        try:
            resp = post_chat(url, model_id, row["system"], max_tokens, timeout)
            text = resp["choices"][0]["message"]["content"]
            return {
                "idx": row["idx"],
                "gold": row["gold"],
                "pred": parse_label(text),
                "gen": text,
                "usage": resp.get("usage", {}),
                "finish_reason": resp["choices"][0].get("finish_reason"),
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            err = str(e)
            time.sleep(1.0 + attempt * 2.0)
    return {
        "idx": row["idx"], "gold": row["gold"], "pred": None,
        "gen": f"[request failed after {retries+1} attempts: {err}]",
        "usage": {}, "finish_reason": "error",
    }


def compute_metrics(preds: list[dict]) -> dict:
    n = len(preds)
    valid = [p for p in preds if p["pred"] is not None]
    n_valid = len(valid)
    if n_valid == 0:
        return {"n": n, "n_valid": 0}

    correct = sum(1 for p in valid if p["pred"] == p["gold"])
    tp = sum(1 for p in valid if p["gold"] == 1 and p["pred"] == 1)
    fp = sum(1 for p in valid if p["gold"] == 0 and p["pred"] == 1)
    fn = sum(1 for p in valid if p["gold"] == 1 and p["pred"] == 0)
    tn = sum(1 for p in valid if p["gold"] == 0 and p["pred"] == 0)

    def safe_div(a: int, b: int) -> float:
        return a / b if b else 0.0

    prec_p = safe_div(tp, tp + fp)
    rec_p = safe_div(tp, tp + fn)
    f1_p = safe_div(2 * prec_p * rec_p, prec_p + rec_p)
    prec_n = safe_div(tn, tn + fn)
    rec_n = safe_div(tn, tn + fp)
    f1_n = safe_div(2 * prec_n * rec_n, prec_n + rec_n)

    return {
        "n": n,
        "n_valid": n_valid,
        "n_unparseable": n - n_valid,
        "accuracy": correct / n_valid,
        "precision_pos": prec_p,
        "recall_pos": rec_p,
        "f1_pos": f1_p,
        "precision_neg": prec_n,
        "recall_neg": rec_n,
        "f1_neg": f1_n,
        "macro_precision": (prec_p + prec_n) / 2,
        "macro_recall": (rec_p + rec_n) / 2,
        "macro_f1": (f1_p + f1_n) / 2,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "gold_pos_count": sum(1 for p in valid if p["gold"] == 1),
        "gold_neg_count": sum(1 for p in valid if p["gold"] == 0),
        "pred_pos_count": sum(1 for p in valid if p["pred"] == 1),
        "pred_neg_count": sum(1 for p in valid if p["pred"] == 0),
    }


def write_markdown(out_md: Path, *, model_id: str, test_file: Path,
                   metrics: dict, n_requested: int, wall_seconds: float,
                   examples: list[dict]) -> None:
    c = metrics.get("confusion", {})
    def pct(x): return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "—"

    md = []
    md.append(f"# SFT Evaluation — `{model_id}`\n")
    md.append(f"- **Test file**: `{test_file}`")
    md.append(f"- **Samples requested**: {n_requested}")
    md.append(f"- **Samples scored**: {metrics.get('n_valid', 0)} "
              f"(unparseable: {metrics.get('n_unparseable', 0)})")
    md.append(f"- **Wall time**: {wall_seconds:.1f}s "
              f"({metrics.get('n', 0) / max(wall_seconds, 1e-6):.2f} req/s)")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| accuracy | **{pct(metrics.get('accuracy', 0))}** |")
    md.append(f"| macro F1 | **{pct(metrics.get('macro_f1', 0))}** |")
    md.append(f"| macro precision | {pct(metrics.get('macro_precision', 0))} |")
    md.append(f"| macro recall | {pct(metrics.get('macro_recall', 0))} |")
    md.append("")
    md.append("## Per-class metrics")
    md.append("")
    md.append("| class | precision | recall | F1 | support (gold) |")
    md.append("|---|---|---|---|---|")
    md.append(f"| **matched (1)** | {pct(metrics.get('precision_pos', 0))} | "
              f"{pct(metrics.get('recall_pos', 0))} | "
              f"{pct(metrics.get('f1_pos', 0))} | "
              f"{metrics.get('gold_pos_count', 0)} |")
    md.append(f"| **not_matched (0)** | {pct(metrics.get('precision_neg', 0))} | "
              f"{pct(metrics.get('recall_neg', 0))} | "
              f"{pct(metrics.get('f1_neg', 0))} | "
              f"{metrics.get('gold_neg_count', 0)} |")
    md.append("")
    md.append("## Confusion matrix")
    md.append("")
    md.append("|              | pred **matched** | pred **not_matched** |")
    md.append("|---|---|---|")
    md.append(f"| gold **matched**     | {c.get('tp', 0)} (TP) | {c.get('fn', 0)} (FN) |")
    md.append(f"| gold **not_matched** | {c.get('fp', 0)} (FP) | {c.get('tn', 0)} (TN) |")
    md.append("")
    md.append("## Distribution sanity")
    md.append("")
    md.append(f"- Gold: matched={metrics.get('gold_pos_count', 0)}, "
              f"not_matched={metrics.get('gold_neg_count', 0)}")
    md.append(f"- Pred: matched={metrics.get('pred_pos_count', 0)}, "
              f"not_matched={metrics.get('pred_neg_count', 0)}")
    md.append("")

    if examples:
        md.append("## Example generations")
        md.append("")
        for e in examples:
            verdict = "✓" if e["pred"] == e["gold"] else "✗"
            md.append(f"### idx={e['idx']}  gold={e['gold']}  pred={e['pred']}  {verdict}")
            md.append("")
            md.append("```")
            md.append(e["gen"].strip())
            md.append("```")
            md.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model-id", required=True,
                   help="Served model name (see curl http://localhost:8000/v1/models).")
    p.add_argument("--test-file", type=Path, default=_REPO / "data" / "sft" / "test.jsonl")
    p.add_argument("--limit", type=int, default=-1,
                   help="-1 = all rows; else take first N for a smoke test.")
    p.add_argument("--concurrency", type=int, default=32,
                   help="Parallel in-flight requests to the server.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--out-md", type=Path, required=True,
                   help="Markdown report output path (e.g. docs/sft_iter1875_eval.md).")
    p.add_argument("--out-jsonl", type=Path, default=None,
                   help="Optional per-row predictions JSONL.")
    p.add_argument("--n-examples", type=int, default=4,
                   help="How many example generations to embed in the report.")
    args = p.parse_args()

    rows = load_test(args.test_file)
    if args.limit > 0:
        rows = rows[:args.limit]
    n = len(rows)
    print(f"[eval] test={args.test_file}  n={n}  model_id={args.model_id}")
    print(f"[eval] url={args.url}  concurrency={args.concurrency}")

    preds: list[dict | None] = [None] * n
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(infer_one, args.url, args.model_id, r,
                      args.max_tokens, args.timeout, args.retries): i
            for i, r in enumerate(rows)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            preds[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                ok = sum(1 for p in preds[:done] if p and p["pred"] == p["gold"])
                scored = sum(1 for p in preds[:done] if p and p["pred"] is not None)
                print(f"[eval] {done}/{n}  rate={rate:.2f} req/s  "
                      f"eta={eta:.0f}s  acc_so_far={ok/max(scored,1):.3f} "
                      f"(parsed {scored})")

    wall = time.time() - t0
    preds_ok = [p for p in preds if p is not None]
    metrics = compute_metrics(preds_ok)

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for p_ in preds_ok:
                f.write(json.dumps(p_, ensure_ascii=False) + "\n")
        print(f"[eval] per-row JSONL → {args.out_jsonl}")

    # Pick a few examples: first correct, first TP, FP, FN, TN.
    examples = []
    seen = set()
    order = [
        ("correct", lambda p: p["pred"] == p["gold"]),
        ("TP", lambda p: p["gold"] == 1 and p["pred"] == 1),
        ("TN", lambda p: p["gold"] == 0 and p["pred"] == 0),
        ("FP", lambda p: p["gold"] == 0 and p["pred"] == 1),
        ("FN", lambda p: p["gold"] == 1 and p["pred"] == 0),
        ("unparseable", lambda p: p["pred"] is None),
    ]
    for _, pred_fn in order:
        for p_ in preds_ok:
            if p_["idx"] in seen:
                continue
            if pred_fn(p_):
                examples.append(p_)
                seen.add(p_["idx"])
                break
        if len(examples) >= args.n_examples:
            break

    write_markdown(args.out_md, model_id=args.model_id,
                   test_file=args.test_file, metrics=metrics,
                   n_requested=n, wall_seconds=wall,
                   examples=examples)
    print(f"[eval] markdown report → {args.out_md}")

    print()
    print("========== summary ==========")
    print(f"n={metrics.get('n',0)} scored={metrics.get('n_valid',0)}  "
          f"unparseable={metrics.get('n_unparseable',0)}")
    print(f"accuracy : {metrics.get('accuracy',0):.4f}")
    print(f"macro F1 : {metrics.get('macro_f1',0):.4f}")
    print(f"F1 pos   : {metrics.get('f1_pos',0):.4f}  "
          f"P={metrics.get('precision_pos',0):.4f} R={metrics.get('recall_pos',0):.4f}")
    print(f"F1 neg   : {metrics.get('f1_neg',0):.4f}  "
          f"P={metrics.get('precision_neg',0):.4f} R={metrics.get('recall_neg',0):.4f}")
    c = metrics.get('confusion', {})
    print(f"confusion: TP={c.get('tp',0)} FP={c.get('fp',0)} "
          f"FN={c.get('fn',0)} TN={c.get('tn',0)}")


if __name__ == "__main__":
    main()
