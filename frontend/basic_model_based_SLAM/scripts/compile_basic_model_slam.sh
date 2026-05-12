#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX_DIR="$REPO_ROOT/docs/els-cas-templates"
TEX_FILE="basic_model_slam_draft.tex"
TEX_BASENAME="${TEX_FILE%.tex}"
TMP_OUT="${TMPDIR:-/tmp}/basic_model_slam_build"
SYNC_OUT="$TEX_DIR/build_output"
BUILD_LOG="$TMP_OUT/${TEX_BASENAME}_build_console.log"

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk not found. Please install the LaTeX toolchain first."
  exit 1
fi

mkdir -p "$TMP_OUT" "$SYNC_OUT"

cd "$TEX_DIR"

# Keep the stable build flow: compile in /tmp, then sync artifacts back.
latexmk -C -outdir="$TMP_OUT" "$TEX_FILE" >/dev/null 2>&1 || true
latexmk -pdf -outdir="$TMP_OUT" -interaction=nonstopmode "$TEX_FILE" | tee "$BUILD_LOG"

for artifact in \
  "$TMP_OUT/$TEX_BASENAME.pdf" \
  "$TMP_OUT/$TEX_BASENAME.log" \
  "$TMP_OUT/$TEX_BASENAME.aux" \
  "$TMP_OUT/$TEX_BASENAME.bbl" \
  "$TMP_OUT/$TEX_BASENAME.blg" \
  "$TMP_OUT/$TEX_BASENAME.fdb_latexmk" \
  "$TMP_OUT/$TEX_BASENAME.fls" \
  "$TMP_OUT/$TEX_BASENAME.out" \
  "$BUILD_LOG"
do
  if [ -f "$artifact" ]; then
    cp -f "$artifact" "$SYNC_OUT"/
  fi
done

echo
echo "Compiled PDF:"
echo "  $SYNC_OUT/$TEX_BASENAME.pdf"
echo "Temporary build directory:"
echo "  $TMP_OUT"
echo "Stable synced output directory:"
echo "  $SYNC_OUT"
