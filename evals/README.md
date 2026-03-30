# mesh-llm Router Evals

A/B comparison of pi agent performance through mesh-llm's multi-model router vs a frontier cloud model.

## Setup

### Mesh (local multi-model)
```bash
# 3 models on M4 Max 52GB (~27GB total, room for KV cache)
MESH_LLM_EPHEMERAL_KEY=1 mesh-llm \
  --model Qwen2.5-32B-Instruct-Q4_K_M \
  --model Qwen2.5-Coder-7B-Instruct-Q4_K_M \
  --model Hermes-2-Pro-Mistral-7B-Q4_K_M
```

Router auto-classifies each request and picks the best model:
- **Qwen2.5-32B** (tier 3) — reasoning, chat, complex code, tool use
- **Qwen2.5-Coder-7B** (tier 2) — code generation/review, fast (85 tok/s)
- **Hermes-7B** (tier 2) — fast chat, simple Q&A (87 tok/s, no tool use)

`MESH_LLM_EPHEMERAL_KEY=1` uses a fresh identity so no external peers connect.

### Cloud baseline
Sonnet via `pi --provider anthropic --model claude-sonnet-4-20250514`.

## Scenarios

Multi-turn conversations that start with chat and progress to tool use:

| Scenario | Turns | What it tests |
|---|---|---|
| **chat-to-code** | 4 | Chat→write code→write tests→review (router must switch models) |
| **debug-session** | 4 | Read files→run code→find/fix bugs→verify (tool-heavy) |
| **edit-file** | 3 | Analyze→multi-step edits→verify (structured editing) |
| **html-app** | 3 | Generate code→validate→iterate (code generation) |
| **explore-repo** | 4 | Bash tools→read files→summarize (repo navigation) |
| **refactor** | 3 | Code review→refactor→verify (code quality) |

## Running

### Multi-turn (recommended — realistic)
```bash
# Single scenario
./evals/run-multi.sh mesh chat-to-code
./evals/run-multi.sh opus chat-to-code

# Compare results
./evals/compare.sh chat-to-code
```

### One-shot (quick, less realistic)
```bash
./evals/run.sh mesh edit-file
./evals/run.sh opus edit-file
```

## Results

Results go to `evals/results/<provider>/<scenario>/`:
- Working files (copied from scenario, edited by agent)
- `_output.txt` — full session capture
- `_screen_turnN.txt` — screen state after each turn
- `_time.txt` — wall clock seconds
- `_turns.txt` — number of turns completed

## What to look for

1. **Correctness** — Did it complete all turns? Are edits right?
2. **Tool use** — Did it use read/edit/bash appropriately?
3. **Routing** — Check `/tmp/mesh-llm-local.log` for which model handled each turn
4. **Speed** — Wall clock per scenario
5. **Model switching** — Does quality degrade when router changes models mid-conversation?
6. **Chat quality** — Are quick chat responses from Hermes comparable to 32B?

## Model capabilities (from testing)

| Model | Tool use | Code gen | Chat | Speed |
|---|---|---|---|---|
| Qwen2.5-32B | ✅ works | ✅ good | ✅ good | ~18 tok/s |
| Qwen2.5-Coder-7B | ✅ works | ✅ great | ⚠️ ok | ~85 tok/s |
| Hermes-7B | ❌ broken | ⚠️ basic | ✅ fast | ~87 tok/s |
| Qwen3-30B-A3B | ❌ thinking format | ✅ good | ❌ empty content | ~22 tok/s |

## Backend benchmark

`evals/backend-benchmark.py` benchmarks local OpenAI-compatible backends serially.

It can compare:

- mesh-managed `llama` via a GGUF model path
- mesh-managed `mlx` via an MLX model directory
- standalone `vllm`
- any custom backend launch command that exposes `/v1/models` and `/v1/chat/completions`

It measures:

- startup time until `/v1/models` is ready
- time to first token
- end-to-end request time
- usage-based completion throughput when reported
- aggregate throughput across one or more concurrency levels

### Simple usage

```bash
python3 evals/backend-benchmark.py \
  --llama-model ~/.models/Qwen2.5-0.5B-Instruct-F16.gguf \
  --mlx-model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/<snapshot> \
  --vllm-model ~/.models/hf/Qwen2.5-0.5B-Instruct \
  --concurrency 1,4,8 \
  --iterations 3
```

### Arbitrary backend combinations

Use a JSON spec file when you want more than the built-in `llama` / `mlx` / `vllm` helpers:

```bash
python3 evals/backend-benchmark.py \
  --spec-file evals/backend-benchmark.example.json \
  --concurrency 1,4,8 \
  --iterations 3
```

The example file shows how to mix mesh-managed backends with standalone commands.
Every backend entry gets its own local port automatically unless you pin one explicitly.

By default the script rejects backend sets whose served model IDs do not normalize to
the same base identity. Pass `--allow-mismatched-models` if you want an intentionally
mixed comparison.
