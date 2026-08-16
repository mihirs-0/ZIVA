#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-14B"
REVISION="40c069824f4251a91eefaf281ebe4c544efd3e18"
IMAGE="vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d"
CONTAINER="ziva-qwen3-14b"

if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "Container ${CONTAINER} already exists; refusing to replace or delete it." >&2
  exit 1
fi

mkdir -p "${HOME}/.cache/huggingface" "${HOME}/.cache/torchinductor" "${HOME}/.triton"

docker run --detach \
  --name "${CONTAINER}" \
  --restart on-failure:10 \
  --gpus all \
  --ipc=host \
  --network=host \
  --user "$(id -u):$(id -g)" \
  --volume /etc/passwd:/etc/passwd:ro \
  --volume /etc/group:/etc/group:ro \
  --volume "${HOME}/.cache:${HOME}/.cache" \
  --volume "${HOME}/.triton:${HOME}/.triton" \
  --env "HOME=${HOME}" \
  --env "HF_HOME=${HOME}/.cache/huggingface" \
  --env "HF_HUB_DISABLE_XET=1" \
  --env "TORCHINDUCTOR_CACHE_DIR=${HOME}/.cache/torchinductor" \
  --env "VLLM_NO_USAGE_STATS=1" \
  "${IMAGE}" \
  "${MODEL}" \
  --revision "${REVISION}" \
  --served-model-name "${MODEL}" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port 8000
