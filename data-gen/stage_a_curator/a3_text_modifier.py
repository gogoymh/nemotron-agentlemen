"""A.3 — Split a cleaned title into `product_name_clean` while preserving the raw.

Real production listings still carry promo noise; we keep both so the model
learns to normalise, but downstream sibling-sampling (C.2) uses the clean
variant for TF-IDF neighbour search.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier
from nemo_curator.stages.text.modifiers.modifier import Modify

from common.io import load_yaml, repo_root, resolve
from common.schema import CLEAN_TEXT, RAW_TEXT

log = logging.getLogger("stage_a.a3")

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "⬀-⯿"
    "]+",
    flags=re.UNICODE,
)


class PromoStripper(DocumentModifier):
    """Removes bracketed promo tags, emoji, and leading marketing keywords."""

    def __init__(
        self,
        strip_patterns: list[str],
        strip_emoji: bool,
        strip_marketing_prefix: list[str],
    ) -> None:
        super().__init__()
        self._patterns = [re.compile(p) for p in strip_patterns]
        self._strip_emoji = strip_emoji
        self._prefix_re = (
            re.compile(rf"^(?:{'|'.join(re.escape(p) for p in strip_marketing_prefix)})[\s:·]*")
            if strip_marketing_prefix
            else None
        )

    def modify_document(self, text: str) -> str:
        out = text
        for p in self._patterns:
            out = p.sub(" ", out)
        if self._strip_emoji:
            out = _EMOJI_RE.sub(" ", out)
        if self._prefix_re:
            for _ in range(3):
                new = self._prefix_re.sub("", out).lstrip()
                if new == out:
                    break
                out = new
        return re.sub(r"\s+", " ", out).strip()


def run(cfg: dict, input_dir: Path | str) -> Path:
    tm = cfg["text_modifier"]
    out_dir = resolve(cfg["io"]["v2_root"]) / "_stage_a" / "a3_modify"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = JsonlReaderStage(file_paths=str(input_dir), fields=None)
    modifier = Modify(
        modifier=PromoStripper(
            strip_patterns=tm["strip_patterns"],
            strip_emoji=tm.get("strip_emoji", True),
            strip_marketing_prefix=tm.get("strip_marketing_prefix", []),
        ),
        input_fields=[RAW_TEXT],
        output_fields=[tm["clean_field"] or CLEAN_TEXT],
    )
    writer = JsonlWriter(path=str(out_dir))

    Pipeline(name="stage_a_modify", stages=[reader, modifier, writer]).run()
    log.info("A.3 text modifier done → %s", out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A.3 text modifier")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/curator.yaml"))
    ap.add_argument("--input-dir", required=True, help="output of A.2")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run(load_yaml(args.config), args.input_dir)


if __name__ == "__main__":
    main()
