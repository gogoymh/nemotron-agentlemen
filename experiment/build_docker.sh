#!/usr/bin/env bash
# Build the two-layer Docker stack for SFT + eval.
#
# Layer 1: megatron-bridge:latest     — Megatron-Bridge/docker/Dockerfile.ci
#   Base: nvcr.io/nvidia/pytorch:26.02-py3 (~25 GB pull, first build ~30-60 min).
# Layer 2: nemotron-agentlemen:latest — experiment/Dockerfile (adds nemo-evaluator).
#
# DISK: the default Docker root /var/lib/docker lives on / which only has
# ~50 GB free on this host. The base image alone is ~25 GB, plus build
# layers will blow past that. Relocate Docker storage to /ephemeral first:
#
#     sudo systemctl stop docker docker.socket
#     sudo mkdir -p /ephemeral/docker
#     sudo rsync -aP /var/lib/docker/ /ephemeral/docker/   # only if non-empty
#     # add "data-root": "/ephemeral/docker" to /etc/docker/daemon.json
#     sudo systemctl start docker
#     docker info | grep "Docker Root Dir"   # verify /ephemeral/docker
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

BASE_TAG="${BASE_TAG:-megatron-bridge:latest}"
FINAL_TAG="${FINAL_TAG:-nemotron-agentlemen:latest}"

if ! git -C "$REPO_ROOT/Megatron-Bridge" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: Megatron-Bridge/ submodule is not checked out."
    echo "Run: git -C $REPO_ROOT submodule update --init --recursive"
    exit 1
fi

echo "[1/2] building $BASE_TAG from Megatron-Bridge/docker/Dockerfile.ci ..."
docker build \
    -f "$REPO_ROOT/Megatron-Bridge/docker/Dockerfile.ci" \
    --target megatron_bridge \
    -t "$BASE_TAG" \
    "$REPO_ROOT/Megatron-Bridge"

echo "[2/2] building $FINAL_TAG (adds nemo-evaluator + pandas) ..."
docker build \
    -f "$REPO_ROOT/experiment/Dockerfile" \
    --build-arg BASE_IMAGE="$BASE_TAG" \
    -t "$FINAL_TAG" \
    "$REPO_ROOT/experiment"

echo "done. images:"
docker image ls --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' \
    | grep -E "^($BASE_TAG|$FINAL_TAG|REPOSITORY)" || true
