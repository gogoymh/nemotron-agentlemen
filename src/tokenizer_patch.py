"""Patch Nemotron-3-Nano's HF chat_template so Megatron computes completion-only loss.

Megatron-Bridge's `_chat_preprocess` (src/megatron/bridge/data/datasets/utils.py)
only produces an assistant-tokens-only `loss_mask` if the chat template contains
`{% generation %}...{% endgeneration %}` markers around assistant content — that
is the HF convention for `apply_chat_template(return_assistant_tokens_mask=True)`.

Nemotron-3-Nano's shipped template does NOT have those markers, so Megatron
silently falls back to `mask = [1] * len(input_ids)` — loss is computed over the
system/user tokens too. For single-turn fixed-prompt SFT, that wastes most of
the gradient signal on re-memorising the prompt.

This module rewrites the assistant emit lines in the template, saves the
modified tokenizer to a directory, and returns that path so callers can point
their tokenizer config at it.
"""
from __future__ import annotations

from pathlib import Path
import re

from transformers import AutoTokenizer


# The exact assistant-content emit lines in Nemotron-3-Nano's chat_template.
# We only wrap the cases that actually emit assistant output — for the
# `<|im_start|>assistant\n<|im_end|>\n` empty-shell case there is nothing
# for the model to learn so we leave it alone.
_PATCHES: list[tuple[str, str]] = [
    # 1) assistant message, no tool_calls, no history-truncation (our SFT path).
    (
        "{{- '<|im_start|>assistant\\n' ~ (content | default('', true) | string | trim) ~ '<|im_end|>\\n' }}",
        "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- (content | default('', true) | string | trim) ~ '<|im_end|>\\n' }}{% endgeneration %}",
    ),
    # 2) assistant message with tool_calls, emits content then <|im_end|> at the end.
    #    Wrap the whole body between `<|im_start|>assistant\n` and `<|im_end|>\n`.
    #    (Kept disabled for now — our test.jsonl/training.jsonl has no tool calls.
    #    If/when tool data is added, uncomment and mirror the simple-branch fix.)
    # 3) Truncation branch (history_thinking rewrite) — same reason.
    (
        "{{- '<|im_start|>assistant\\n' ~ c ~ '<|im_end|>\\n' }}",
        "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- c ~ '<|im_end|>\\n' }}{% endgeneration %}",
    ),
]


def patch_chat_template(template: str) -> str:
    out = template
    applied = 0
    for old, new in _PATCHES:
        if old in out:
            out = out.replace(old, new, 1)
            applied += 1
    if applied == 0:
        raise RuntimeError(
            "No assistant emit line matched — chat_template may have changed. "
            "Inspect the template and update _PATCHES before retrying."
        )
    return out


def save_patched_tokenizer(src: str, out_dir: Path) -> Path:
    """Load `src` (HF repo id or local path), rewrite chat_template, save to `out_dir`.

    Returns the output dir (as absolute Path) so callers can pass it to the
    recipe's `TokenizerConfig(tokenizer_model=...)`.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if not tok.chat_template:
        raise RuntimeError(f"{src} tokenizer has no chat_template")

    if re.search(r"\{%-?\s*generation\s*-?%\}", tok.chat_template):
        # Already has it; just re-save as-is.
        tok.save_pretrained(out_dir)
        return out_dir

    tok.chat_template = patch_chat_template(tok.chat_template)
    tok.save_pretrained(out_dir)
    return out_dir


def verify_mask(tokenizer_dir: Path, sample_messages: list[dict]) -> dict:
    """Sanity check: tokenize `sample_messages` with return_assistant_tokens_mask=True
    and make sure (a) we get a non-empty mask, (b) mask has at least one 1 AND one 0.
    """
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    encoded = tok.apply_chat_template(
        sample_messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = encoded.get("assistant_masks")
    input_ids = encoded["input_ids"]
    if not mask:
        raise RuntimeError("assistant_masks missing — patch did not take effect")
    n_one = sum(1 for x in mask if x == 1)
    n_zero = sum(1 for x in mask if x == 0)
    if n_one == 0 or n_zero == 0:
        raise RuntimeError(
            f"degenerate mask: {n_one} ones / {n_zero} zeros — check _PATCHES."
        )
    return {
        "n_tokens": len(input_ids),
        "n_assistant": n_one,
        "n_context": n_zero,
        "assistant_ratio": n_one / len(input_ids),
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--src", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--verify-with", type=Path, default=None,
                   help="Optional: path to a JSONL. First row is used to sanity-check "
                        "the mask.")
    args = p.parse_args()

    out = save_patched_tokenizer(args.src, args.out_dir)
    print(f"[tokenizer-patch] saved patched tokenizer → {out}")

    if args.verify_with:
        with open(args.verify_with, encoding="utf-8") as f:
            first = json.loads(f.readline())
        stats = verify_mask(out, first["messages"])
        print(
            f"[tokenizer-patch] verify: n_tokens={stats['n_tokens']} "
            f"assistant={stats['n_assistant']} context={stats['n_context']} "
            f"assistant_ratio={stats['assistant_ratio']:.3f}"
        )
