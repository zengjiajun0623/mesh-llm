# MoE Strategy Benchmarks

This document summarizes the offline MoE strategy benchmark suite added in this branch.

Models tested on `studio54.local`:

- `GLM-4.7-Flash-Q4_K_M`
- `Qwen3-Coder-Next-Q4_K_M`

The suite compares four questions:

1. Which ranking source best matches full `llama-moe-analyze`?
2. How much does ranking quality matter more than grouping shape?
3. Can a short `micro-analyze` replace a full analyze pass?
4. Which live CLI knobs should we expose to test these strategies in mesh runs?

## Bottom Line

- Gold standard: full `llama-moe-analyze`
- Best practical fallback: `micro-analyze` with `--all-layers`
- Best weight-only heuristic in this branch: `heuristic-max`
- Current safe zero-analysis fallback: `sequential`

The strongest result is that `micro-analyze` is dramatically better than any weight-only heuristic on both models, while still costing much less than a full analyze run.

## Ranking Results

### GLM-4.7-Flash-Q4_K_M

- Full analyze remains the reference ranking.
- `micro-1p-8t-all-layers` matched full analyze exactly:
  - Spearman: `1.00`
  - Recall@24: `1.00`
  - Weighted recall@24: `1.00`
  - Runtime: `17.29s`
- `heuristic-max` was the best weight-only heuristic, but still poor:
  - Spearman: `0.06`
  - Recall@24: `0.46`
  - Weighted recall@24: `0.27`
- All tested heuristics missed expert `0`, which is unacceptable for this model because full analyze shows expert `0` carries `22.94%` of gate mass.

### Qwen3-Coder-Next-Q4_K_M

- `micro-1p-8t-all-layers` was already close to full analyze:
  - Spearman: `0.951`
  - Recall@256: `0.930`
  - Weighted recall@256: `0.966`
  - Runtime: `32.09s`
- `micro-4p-32t-all-layers` matched full analyze exactly:
  - Spearman: `1.00`
  - Recall@256: `1.00`
  - Weighted recall@256: `1.00`
  - Runtime: `314.95s`
- `heuristic-max` again beat the other weight-only heuristics, but stayed well below micro-analyze:
  - Spearman: `0.020`
  - Recall@256: `0.516`
  - Weighted recall@256: `0.741`

## Grouping Results

With a good ranking, both grouping shapes work:

- `current-analyze` is already strong.
- `snake-analyze-replicated` is slightly better balanced.

With a bad ranking, grouping does not rescue quality:

- On GLM, `snake-heuristic-replicated` is materially worse than `current-sequential`.
- On Qwen, heuristic snake-draft is usable but still clearly below analyze-backed grouping.

Practical interpretation:

- Ranking quality matters more than grouping shape.
- `snake-draft` is worth testing live, but only when paired with a good ranking source.

## Analysis Cost

Startup cost by strategy:

- `bundled / cached analyze`: local config or CSV read only; no `llama-moe-analyze` process
- `sequential`: GGUF header read only; no ranking analysis
- `heuristic-*`: GGUF tensor scan to score router weights; no `llama-moe-analyze`
- `micro-analyze`: short `llama-moe-analyze` run
- `analyze`: full `llama-moe-analyze` run

### Full `llama-moe-analyze`

Timed on `studio54.local` with:

```bash
/usr/bin/time -lp ./llama-moe-analyze -m MODEL --all-layers --export-ranking /tmp/ranking.csv -n 32 -c 4096 -ngl 99
```

- `GLM-4.7-Flash-Q4_K_M`: `44.27s`
- `Qwen3-Coder-Next-Q4_K_M`: `106.74s`

### Micro analyze

Measured from the benchmark suite:

- GLM `micro-1p-8t-all-layers`: `17.29s`
- Qwen `micro-1p-8t-all-layers`: `32.09s`

### Bundled / cached ranking

- Catalog ranking and cached CSV loading are effectively startup-time file reads.
- They do not launch `llama-moe-analyze`.
- This is the cheapest path, but only as good as the bundled or cached artifact.

### Sequential / heuristic startup

- `sequential` only reads GGUF metadata to detect expert counts, then uses `0..N`.
- `heuristic-*` scans router tensors from the GGUF and computes a ranking locally.
- Neither path launches `llama-moe-analyze`, but both are weaker than `micro-analyze` on the tested models.

## Recommendations

Default behavior should stay conservative for now:

- Keep `auto` as the current stable behavior.
- Prefer `micro-analyze` when we explicitly want a better fallback than sequential.
- Do not make the current weight-only heuristic the default fallback yet.

If we change the default later, this benchmark suggests:

1. `bundled / cached analyze`
2. `micro-analyze --all-layers`
3. `sequential`
4. weight-only heuristics

## Benchmark Commands

Import a small fixed corpus:

```bash
mesh-llm benchmark import-prompts \
  --source mt-bench \
  --limit 8 \
  --max-tokens 256 \
  --output evals/moe/prompts/mt-bench-8.jsonl
```

Run the full offline suite:

```bash
mesh-llm benchmark moe-model-matrix \
  --model /Volumes/External/models/GLM-4.7-Flash-Q4_K_M.gguf \
  --model /Volumes/External/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf \
  --nodes 2 \
  --prompts evals/moe/prompts/mt-bench-8.jsonl \
  --output /tmp/moe-model-matrix.json
```

Run individual slices:

```bash
mesh-llm benchmark moe-heuristic --model /path/to/model.gguf
mesh-llm benchmark moe-grouping --model /path/to/model.gguf --nodes 2
mesh-llm benchmark moe-micro-analyze --model /path/to/model.gguf --prompts evals/moe/prompts/mt-bench-8.jsonl
```

## Live Runtime Examples

These new flags are meant for live MoE split experiments:

Full analyze before split:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking analyze \
  --moe-grouping shared-core
```

Micro analyze before split:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking micro-analyze \
  --moe-micro-prompt-count 1 \
  --moe-micro-tokens 8 \
  --moe-micro-layers all \
  --moe-grouping shared-core
```

Heuristic max + snake draft:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking heuristic-max \
  --moe-grouping snake-draft \
  --moe-replicate 256
```

Sequential fallback + shared core overlap:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking sequential \
  --moe-grouping shared-core \
  --moe-overlap 1
```
