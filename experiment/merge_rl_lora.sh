#!/usr/bin/env bash
# Merge a TRL/PEFT LoRA checkpoint (from experiment/rl_from_sft_trl.sh) into
# the base SFT model and export as HF bf16 safetensors so it can be served
# by vLLM for eval.
#
# Direct safetensors merge — doesn't instantiate the Nemotron-H model class
# (which would pull in mamba_ssm / Mamba kernels). Just reads safetensors,
# applies  W += (B @ A) * (alpha / r)  per LoRA target, and writes back.
#
# Inputs:
#   RL_CKPT     : PEFT checkpoint dir  (adapter_config.json + adapter_model.safetensors)
#                 default: artifacts/rl/nemotron-30b-trl/checkpoints/checkpoint-$STEP
#   BASE_MODEL  : base HF model path (what adapter_config.json references)
#                 default: artifacts/sft/nemotron-30b/hf-iter_0001875
#   HF_OUT      : output dir for merged HF model (bf16 safetensors, ~59 GB)
#                 default: /mnt/data/artifacts/rl/nemotron-30b-trl/hf-checkpoint-$STEP
#
# Usage:
#   STEP=200 bash experiment/merge_rl_lora.sh
#   RL_CKPT=... BASE_MODEL=... HF_OUT=... bash experiment/merge_rl_lora.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano}"

STEP="${STEP:-200}"
RL_CKPT="${RL_CKPT:-$REPO_ROOT/artifacts/rl/nemotron-30b-trl/checkpoints/checkpoint-$STEP}"
BASE_MODEL="${BASE_MODEL:-$REPO_ROOT/artifacts/sft/nemotron-30b/hf-iter_0001875}"
HF_OUT="${HF_OUT:-/mnt/data/artifacts/rl/nemotron-30b-trl/hf-checkpoint-$STEP}"

if [[ ! -f "$RL_CKPT/adapter_config.json" ]]; then
    echo "ERROR: PEFT adapter not found at $RL_CKPT" >&2
    exit 1
fi
if [[ ! -f "$BASE_MODEL/config.json" ]]; then
    echo "ERROR: base model not found at $BASE_MODEL" >&2
    exit 1
fi

mkdir -p "$HF_OUT"

ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

RL_CKPT_C="${RL_CKPT/#$REPO_ROOT/\/workdir}"
BASE_MODEL_C="${BASE_MODEL/#$REPO_ROOT/\/workdir}"

echo "[merge-rl] adapter  = $RL_CKPT"
echo "[merge-rl] base     = $BASE_MODEL"
echo "[merge-rl] out      = $HF_OUT"

# Write the merge script to a tmp file in the repo (mounted at /workdir).
TMP_PY="$REPO_ROOT/.tmp_merge_rl_lora.py"
cat >"$TMP_PY" <<'PYEOF'
"""Pure-safetensors LoRA merge. Avoids transformers model classes so we
don't need mamba_ssm / Nemotron-H trust_remote_code."""
import json, os, pathlib, re, shutil, sys
from collections import defaultdict
import torch
from safetensors.torch import load_file, save_file

adapter = os.environ["ADAPTER_DIR"]
base    = os.environ["BASE_DIR"]
out     = os.environ["OUT_DIR"]

adapter_cfg = json.loads((pathlib.Path(adapter) / "adapter_config.json").read_text())
r = adapter_cfg["r"]
alpha = adapter_cfg["lora_alpha"]
targets = adapter_cfg["target_modules"]
scaling = alpha / r
print(f"[merge-rl] r={r} alpha={alpha} scaling={scaling} targets={targets}", flush=True)

# Load adapter weights (usually ~10 MB total, tiny).
lora_file = pathlib.Path(adapter) / "adapter_model.safetensors"
lora_raw = load_file(str(lora_file))
print(f"[merge-rl] adapter keys: {len(lora_raw)}", flush=True)

# Group by module path. PEFT saves as base_model.model.<path>.lora_[AB].weight
# (plus an optional base_layer.weight mirror that we skip).
lora_pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
key_re = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$")
for k, v in lora_raw.items():
    m = key_re.match(k)
    if not m:
        continue
    path = m.group(1)
    lora_pairs[path][m.group(2)] = v

# Precompute deltas, [out, in] bf16.
deltas: dict[str, torch.Tensor] = {}
for path, pair in lora_pairs.items():
    if "A" not in pair or "B" not in pair:
        print(f"[merge-rl] WARN incomplete pair for {path}", flush=True)
        continue
    A = pair["A"].float()  # [r, in]
    B = pair["B"].float()  # [out, r]
    deltas[f"{path}.weight"] = ((B @ A) * scaling).to(torch.bfloat16)
print(f"[merge-rl] computed {len(deltas)} deltas", flush=True)

# Load base index.
index_path = pathlib.Path(base) / "model.safetensors.index.json"
if index_path.exists():
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
else:
    # Single-file base.
    weight_map = None

out_p = pathlib.Path(out)
out_p.mkdir(parents=True, exist_ok=True)

applied: set[str] = set()
if weight_map is None:
    shard_path = pathlib.Path(base) / "model.safetensors"
    w = load_file(str(shard_path))
    for k, d in deltas.items():
        if k in w:
            w[k] = w[k].to(torch.bfloat16) + d
            applied.add(k)
    save_file(w, str(out_p / "model.safetensors"), metadata={"format": "pt"})
    print("[merge-rl] wrote model.safetensors", flush=True)
else:
    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for k, shard in weight_map.items():
        shard_to_keys[shard].append(k)
    for shard_name, keys in shard_to_keys.items():
        shard_path = pathlib.Path(base) / shard_name
        w = load_file(str(shard_path))
        hits = 0
        for k in keys:
            if k in deltas:
                w[k] = w[k].to(torch.bfloat16) + deltas[k]
                applied.add(k)
                hits += 1
        save_file(w, str(out_p / shard_name), metadata={"format": "pt"})
        print(f"[merge-rl] wrote {shard_name}  (merged {hits}/{len(keys)} tensors)", flush=True)

missing = set(deltas) - applied
if missing:
    print(f"[merge-rl] WARN {len(missing)} deltas did not match any base key", flush=True)
    for k in list(missing)[:5]:
        print(f"           e.g. {k}", flush=True)
    sys.exit(2)
print(f"[merge-rl] applied {len(applied)} deltas", flush=True)

# Copy non-weight metadata from base, overlay tokenizer from adapter.
for fn in ["model.safetensors.index.json", "config.json", "generation_config.json"]:
    src = pathlib.Path(base) / fn
    if src.exists():
        shutil.copy2(src, out_p / fn)
# Remote-code model files — config.json's auto_map points at these by name.
for py in pathlib.Path(base).glob("*.py"):
    shutil.copy2(py, out_p / py.name)
for fn in ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
           "chat_template.jinja", "added_tokens.json"]:
    src = pathlib.Path(adapter) / fn
    if src.exists():
        shutil.copy2(src, out_p / fn)
    else:
        src = pathlib.Path(base) / fn
        if src.exists():
            shutil.copy2(src, out_p / fn)

# vLLM's NemotronH loader reads rms_norm_eps; alias it.
cfg_path = out_p / "config.json"
cfg = json.loads(cfg_path.read_text())
if "rms_norm_eps" not in cfg:
    cfg["rms_norm_eps"] = cfg.get("layer_norm_epsilon", 1e-5)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[merge-rl] patched rms_norm_eps = {cfg['rms_norm_eps']}", flush=True)

print("[merge-rl] done", flush=True)
PYEOF

# CPU-only merge: no forward pass, just tensor arithmetic on weights.
# Runtime is bounded by disk IO (read 59 GB base, write 59 GB merged).
docker run --rm --ipc=host --shm-size=16g \
    -v "$REPO_ROOT":/workdir \
    -v /mnt/data:/mnt/data \
    -e HF_HOME=/workdir/.cache/hf \
    -e HF_HUB_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES="" \
    -e ADAPTER_DIR="$RL_CKPT_C" \
    -e BASE_DIR="$BASE_MODEL_C" \
    -e OUT_DIR="$HF_OUT" \
    "${ENV_FILE_ARG[@]}" \
    -w /workdir \
    "$IMAGE" \
    /opt/nemo_rl_venv/bin/python /workdir/.tmp_merge_rl_lora.py

rm -f "$TMP_PY"

echo "done — merged HF model at $HF_OUT"
