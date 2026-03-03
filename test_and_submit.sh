#!/usr/bin/env bash
#
# Grading script for LLM Efficiency (KV cache + LoRA).
# Runs tests, demo scripts, and benchmark, saving all output to results/.
#
# Usage:  bash test_and_submit.sh
#

set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="results/${TIMESTAMP}"

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  LLM Efficiency — Grading"
echo "  Date    : $(date)"
echo "========================================"
echo ""

# ── 1. KV cache tests ──────────────────────────────────────────────

echo ">> Running KV cache tests ..."
uv run --extra dev pytest tests/test_kv_cache.py -v --tb=short 2>&1 | tee "$RESULTS_DIR/test_kv_cache.txt"
echo ""

# ── 2. LoRA tests ──────────────────────────────────────────────────

echo ">> Running LoRA tests ..."
uv run --extra dev pytest tests/test_lora.py -v --tb=short 2>&1 | tee "$RESULTS_DIR/test_lora.txt"
echo ""

# ── 3. KV cache demo sort ─────────────────────────────────────────

echo ">> Running kv_cache/demo_sort_kv.py ..."
uv run kv_cache/demo_sort_kv.py 2>&1 | tee "$RESULTS_DIR/demo_sort_kv.txt"
echo ""

# ── 4. KV cache benchmark ─────────────────────────────────────────

echo ">> Running kv_cache/benchmark.py (this may take a few minutes) ..."
uv run kv_cache/benchmark.py 2>&1 | tee "$RESULTS_DIR/benchmark_kv.txt"
echo ""

# ── 5. LoRA demo sort ─────────────────────────────────────────────

echo ">> Running lora/demo_sort_lora.py ..."
uv run lora/demo_sort_lora.py 2>&1 | tee "$RESULTS_DIR/demo_sort_lora.txt"
echo ""

# ── Summary ────────────────────────────────────────────────────────

echo "========================================"
echo "  All done! Results saved in: $RESULTS_DIR/"
echo "========================================"
ls -l "$RESULTS_DIR/"
