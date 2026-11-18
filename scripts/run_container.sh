#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ASSET_ROOT:?Set ASSET_ROOT to the absolute external asset directory}"
ASSET_ROOT="$(realpath -e -- "$ASSET_ROOT")"
test -d "$ASSET_ROOT"
mkdir -p "$ROOT/runs"
exec docker run --rm --gpus all --ipc=host --network=host   --user "$(id -u):$(id -g)"   --env HOME=/tmp --env MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29519}"   --mount "type=bind,src=$ROOT,dst=/aligned,readonly"   --mount "type=bind,src=$ROOT/runs,dst=/aligned/runs"   --mount "type=bind,src=$ASSET_ROOT,dst=/assets,readonly"   --workdir /aligned/FastWAM "${IMAGE:-fastwam-h20:ngc2509}"   bash /aligned/scripts/run_100.sh
