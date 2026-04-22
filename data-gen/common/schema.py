"""Canonical column names shared across Stage A (Curator) and Stage C (DataDesigner).

Keeping these in one place avoids silent schema drift between the raw pool,
the curated pool, and the synthetic pools — they all have to round-trip into
`dataset/v2/sft/*.jsonl` with consistent field names.
"""
from __future__ import annotations

# ---- Raw unlabeled listing (dataset/get_data/raw.jsonl) -------------------
RAW_ID = "listing_id"
RAW_TEXT = "product_name"        # original noisy title
RAW_BRAND = "brand"
RAW_PLATFORM = "platform"

# ---- Stage A additions ----------------------------------------------------
CLEAN_TEXT = "product_name_clean"    # A.3 emits this; raw title is preserved
QUALITY_SCORE = "quality_score"      # A.2 fastText score (0..1, hq probability)
DEDUP_CLUSTER = "_fuzzy_cluster_id"  # A.1 bookkeeping, dropped before serialize

# ---- Labeled / synthetic pair schema (shared with dataset/v2/sft/*) -------
P1_NAME = "p1_name"
P2_NAME = "p2_name"
LABEL = "label"                  # human ground truth: "matched" | "not_matched"
DECISION = "decision"            # gpt-5-mini decision in labeled; may differ
REASON = "reason"                # evidence / chain of thought
CONFIDENCE = "confidence"        # High | Med | Low (C.2 pseudo-labels)
SOURCE = "source"                # labeled | regenerated | pseudo | synthetic_hard_{neg,pos}

# Mirrors src/config.py so downstream SFT builder parses without adjustment.
LABEL_MATCHED = "1"
LABEL_NOT_MATCHED = "0"
DECISION_TO_LABEL = {"matched": LABEL_MATCHED, "not_matched": LABEL_NOT_MATCHED}

LABEL_RE = r"<label>\s*([01])\s*</label>"
REASON_RE = r"<reason>(.*?)</reason>"
