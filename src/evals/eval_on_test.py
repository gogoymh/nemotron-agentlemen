"""Standalone evaluation on data/sft/test.jsonl.

Takes a HF-format model (either the base model for a baseline, or an SFT
checkpoint merged back to HF via `examples/peft/merge_lora.py`), runs batched
greedy inference over `data/sft/test.jsonl`, parses `<label>0|1</label>` from
each generation, and reports accuracy / precision / recall / F1 for the
positive (matched=1) class plus a confusion matrix.

Uses vLLM for throughput — 1000 samples on 8×H100 PCIe (tp=1, dp implied)
finishes in a few minutes. Falls back to transformers if vLLM is not
importable.

Usage (from repo root, inside the NeMo container or any env with vLLM):

    # baseline (no SFT)
    python -m src.evals.eval_on_test \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --out-dir results/eval/baseline

    # after SFT: first merge LoRA → HF, then point --model at the merged dir
    python examples/peft/merge_lora.py \
        --hf-model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --lora-checkpoint artifacts/sft/nemotron-30b/checkpoints/iter_0003500 \
        --output artifacts/sft/nemotron-30b/merged
    python -m src.evals.eval_on_test \
        --model artifacts/sft/nemotron-30b/merged \
        --out-dir results/eval/sft
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

LABEL_RE = re.compile(r"<label>\s*([01])\s*</label>")


def load_test(path: Path) -> list[dict]:
    """Pull (system_prompt, gold_label) from each chat row.

    The SFT format stores one assistant turn; its `<label>` tag is the gold.
    Rows without a parseable label are dropped (the prepare_data step should
    never emit those, but we don't trust silently).
    """
    rows = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            sys_msg = next((m for m in d["messages"] if m["role"] == "system"), None)
            ast_msg = next((m for m in d["messages"] if m["role"] == "assistant"), None)
            if sys_msg is None or ast_msg is None:
                skipped += 1
                continue
            match = LABEL_RE.search(ast_msg["content"])
            if not match:
                skipped += 1
                continue
            rows.append({
                "idx": i,
                "system": sys_msg["content"],
                "gold_label": int(match.group(1)),
                "gold_full": ast_msg["content"],
            })
    if skipped:
        print(f"[eval] skipped {skipped} malformed rows")
    return rows


def parse_label(text: str) -> int | None:
    m = LABEL_RE.search(text)
    return int(m.group(1)) if m else None


def build_prompts(rows, tokenizer) -> list[str]:
    """Render the chat template with only the system message + a generation
    prefix — the model has to produce the assistant turn from scratch."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": r["system"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]


def generate_vllm(model: str, prompts, rows, max_tokens: int, tp: int, dtype: str,
                  print_every: int, print_full: bool):
    from vllm import LLM, SamplingParams
    print(f"[eval] loading via vLLM (tp={tp}, dtype={dtype})")
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        dtype=dtype,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        temperature=0.0,        # greedy — eval must be deterministic
        max_tokens=max_tokens,
        stop=["</label>"],      # cut off after the tag; label itself stays in
        include_stop_str_in_output=True,
    )
    t0 = time.time()
    out = llm.generate(prompts, sampling)
    dt = time.time() - t0
    print(f"[eval] generated {len(out)} responses in {dt:.1f}s "
          f"({len(out)/max(dt,1e-6):.2f} req/s)")
    texts = [o.outputs[0].text for o in out]
    if print_every > 0:
        from tqdm.auto import tqdm as _tqdm
        for i, (r, t) in enumerate(_tqdm(list(zip(rows, texts)),
                                          desc="vllm post-print", unit="req")):
            if i % print_every == 0:
                _print_sample(i, r["gold_label"], parse_label(t), t, print_full,
                              writer=_tqdm.write)
    return texts


def _print_sample(idx: int, gold, pred, gen, full: bool, writer=print) -> None:
    writer(f"\n── sample #{idx}  gold={gold}  pred={pred}{'  ✓' if pred == gold else '  ✗'}")
    shown = gen if full else (gen if len(gen) < 600 else gen[:300] + f"  …<{len(gen)-600} chars>…  " + gen[-300:])
    writer(shown)
    writer("─" * 60)


def generate_hf(model: str, prompts, rows, max_tokens: int, batch_size: int,
                dtype: str, print_every: int, print_full: bool,
                gpu: int, multi_gpu: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm.auto import tqdm

    # Single-GPU is a lot faster for inference — no cross-device activation
    # shuffling per forward pass. 30B bf16 ≈ 60GB fits on 1× H100 80GB.
    if multi_gpu:
        device_map = "auto"
        placement = "multi-gpu (device_map=auto)"
    else:
        device_map = {"": f"cuda:{gpu}"}
        placement = f"single-gpu (cuda:{gpu})"
    print(f"[eval] loading via transformers (dtype={dtype}, batch={batch_size}, {placement})")
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model,
        dtype=getattr(torch, dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    mdl.eval()
    outs = []
    done = 0
    correct = 0
    scored = 0
    pbar = tqdm(total=len(prompts), desc="hf gen", unit="req", smoothing=0.05)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )
        in_lens = enc.attention_mask.sum(dim=1).tolist()
        for row_ids, in_len in zip(gen, in_lens):
            text = tok.decode(row_ids[in_len:], skip_special_tokens=True)
            outs.append(text)
            pred = parse_label(text)
            gold = rows[done]["gold_label"]
            if pred is not None:
                scored += 1
                if pred == gold:
                    correct += 1
            if print_every > 0 and done % print_every == 0:
                _print_sample(done, gold, pred, text, print_full, writer=pbar.write)
            done += 1
            pbar.update(1)
            pbar.set_postfix(acc=f"{correct/scored:.3f}" if scored else "—",
                             parsed=f"{scored}/{done}")
    pbar.close()
    return outs


def compute_metrics(preds: list[dict]) -> dict:
    n = len(preds)
    valid = [p for p in preds if p["pred"] is not None]
    n_valid = len(valid)
    if not valid:
        return {"n": n, "n_valid": 0, "accuracy": 0.0}

    correct = sum(1 for p in valid if p["pred"] == p["gold"])
    tp = sum(1 for p in valid if p["gold"] == 1 and p["pred"] == 1)
    fp = sum(1 for p in valid if p["gold"] == 0 and p["pred"] == 1)
    fn = sum(1 for p in valid if p["gold"] == 1 and p["pred"] == 0)
    tn = sum(1 for p in valid if p["gold"] == 0 and p["pred"] == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Macro-F1 across both classes (neg-class precision/recall flipped).
    prec_neg = tn / (tn + fn) if (tn + fn) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) else 0.0
    macro_f1 = (f1 + f1_neg) / 2

    return {
        "n": n,
        "n_valid": n_valid,
        "n_unparseable": n - n_valid,
        "accuracy": correct / n_valid,
        "precision_pos": prec,
        "recall_pos": rec,
        "f1_pos": f1,
        "precision_neg": prec_neg,
        "recall_neg": rec_neg,
        "f1_neg": f1_neg,
        "macro_f1": macro_f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF repo id OR local dir with HF safetensors + tokenizer.")
    p.add_argument("--test-file", type=Path, default=_REPO / "data" / "sft" / "test.jsonl")
    p.add_argument("--out-dir", type=Path, default=_REPO / "results" / "eval")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--limit", type=int, default=-1,
                   help="-1 = all rows; else take the first N rows for a quick sanity check.")
    p.add_argument("--row-idx", type=int, default=None,
                   help="Run on exactly ONE row at the given 0-based index "
                        "(overrides --limit). Implies --print-every 1 --print-full.")
    p.add_argument("--one", action="store_true",
                   help="Shortcut for --row-idx 0 (infer just the first sample).")
    p.add_argument("--backend", choices=["vllm", "hf", "auto"], default="auto")
    p.add_argument("--hf-batch-size", type=int, default=4,
                   help="Only used when backend=hf")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index for HF backend (default: 0). "
                        "Inference on a single GPU is much faster than "
                        "device_map=auto when the model fits.")
    p.add_argument("--multi-gpu", action="store_true",
                   help="HF backend: shard the model across all visible GPUs "
                        "(device_map=auto). Slower than single-GPU when the "
                        "model fits on one card; use only if OOM on --gpu.")
    p.add_argument("--print-every", type=int, default=0,
                   help="Print every Nth generated response to stdout "
                        "(0 = off). Good for live inspection of decoding.")
    p.add_argument("--print-full", action="store_true",
                   help="When --print-every > 0, print the full generation "
                        "instead of a head+tail snippet.")
    args = p.parse_args()

    rows = load_test(args.test_file)

    # Single-row mode: --one (idx 0) or --row-idx N picks exactly one sample.
    if args.one and args.row_idx is None:
        args.row_idx = 0
    if args.row_idx is not None:
        if not (0 <= args.row_idx < len(rows)):
            raise SystemExit(f"--row-idx {args.row_idx} out of range (0..{len(rows)-1})")
        rows = [rows[args.row_idx]]
        # Force verbose single-sample print when user asked for one row.
        if args.print_every == 0:
            args.print_every = 1
        args.print_full = True
        print(f"[eval] single-row mode: idx={args.row_idx}")
    elif args.limit > 0:
        rows = rows[:args.limit]

    print(f"[eval] model={args.model}")
    print(f"[eval] test={args.test_file}  n={len(rows)}")

    # Tokenizer is needed for chat template regardless of backend.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = build_prompts(rows, tokenizer)

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except ImportError:
            backend = "hf"
    print(f"[eval] backend={backend}")

    if backend == "vllm":
        gens = generate_vllm(args.model, prompts, rows, args.max_tokens, args.tp,
                             args.dtype, args.print_every, args.print_full)
    else:
        gens = generate_hf(args.model, prompts, rows, args.max_tokens,
                           args.hf_batch_size, args.dtype,
                           args.print_every, args.print_full,
                           args.gpu, args.multi_gpu)

    preds = []
    for r, gen in zip(rows, gens):
        preds.append({
            "idx": r["idx"],
            "gold": r["gold_label"],
            "pred": parse_label(gen),
            "gen": gen,
            "gold_full": r["gold_full"],
        })

    metrics = compute_metrics(preds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p_ in preds:
            f.write(json.dumps(p_, ensure_ascii=False) + "\n")
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, **metrics}, f, indent=2, ensure_ascii=False)

    print("\n========== results ==========")
    print(f"  n={metrics['n']}  parsed={metrics['n_valid']}  "
          f"unparseable={metrics.get('n_unparseable', 0)}")
    print(f"  accuracy     : {metrics['accuracy']:.4f}")
    print(f"  F1 (matched) : {metrics['f1_pos']:.4f}   "
          f"P={metrics['precision_pos']:.4f}  R={metrics['recall_pos']:.4f}")
    print(f"  F1 (unmatch) : {metrics['f1_neg']:.4f}   "
          f"P={metrics['precision_neg']:.4f}  R={metrics['recall_neg']:.4f}")
    print(f"  macro F1     : {metrics['macro_f1']:.4f}")
    c = metrics["confusion"]
    print(f"  confusion    : TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}")
    print(f"\nper-row preds : {args.out_dir / 'predictions.jsonl'}")
    print(f"summary       : {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
