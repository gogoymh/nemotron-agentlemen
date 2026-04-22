# Layer nemo-evaluator on top of the Megatron-Bridge CI image so one image
# serves both SFT (torchrun src/train_sft.py) and eval (nemo-evaluator BYOB).
#
# Build via experiment/build_docker.sh (expects megatron-bridge:latest to exist).
ARG BASE_IMAGE=megatron-bridge:latest
FROM ${BASE_IMAGE}

# /opt/venv is the UV_PROJECT_ENVIRONMENT pinned by Dockerfile.ci.
RUN /opt/venv/bin/pip install --no-cache-dir \
        "nemo-evaluator" \
        pandas

WORKDIR /workspace/project
