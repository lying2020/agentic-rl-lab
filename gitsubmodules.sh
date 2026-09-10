#!/usr/bin/env bash
# One-click init / update for all git submodules listed in .gitmodules.
# Usage (from anywhere):
#   ./gitsubmodules.sh
#   ./gitsubmodules.sh --remote          # also fast-forward to tracked remote branches
#   ./gitsubmodules.sh algorithm/verl    # only update selected paths
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f .gitmodules ]]; then
  echo "error: .gitmodules not found in $ROOT" >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  echo "error: $ROOT is not a git repository" >&2
  exit 1
fi

REMOTE=0
PATHS=()
for arg in "$@"; do
  case "$arg" in
    --remote|-r) REMOTE=1 ;;
    -h|--help)
      sed -n '2,7p' "$0"
      exit 0
      ;;
    *) PATHS+=("$arg") ;;
  esac
done

echo "==> sync submodule URLs from .gitmodules"
git submodule sync --recursive

echo "==> fetch / checkout submodules"
UPDATE_ARGS=(update --init --recursive --jobs "$(nproc 2>/dev/null || echo 4)")
if [[ "$REMOTE" -eq 1 ]]; then
  UPDATE_ARGS+=(--remote)
fi
if [[ ${#PATHS[@]} -gt 0 ]]; then
  git submodule "${UPDATE_ARGS[@]}" -- "${PATHS[@]}"
else
  git submodule "${UPDATE_ARGS[@]}"
fi

echo
echo "==> submodule status"
git submodule status --recursive
echo
echo "done."
