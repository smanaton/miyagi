<p align="center">
  <img src="https://github.com/Apothic-AI/miyagi/blob/master/assets/miyagi-logo.png?raw=true" alt="Miyagi" width="360">
</p>

# Miyagi

Miyagi is a Rust toolkit for sparse XOR adaptation of true-binary GGUF language
models. It discovers supported Qwen/Bonsai Q1_0 tensors, evaluates selected
token preferences, searches for reversible row-level patches, and measures the
result against probes or generation datasets.

Miyagi is built on the sibling [`wwama`](https://github.com/Apothic-AI/wwama) crate. `wwama` owns GGUF
loading, tokenization, inference, generation, tensor descriptors, backend
transfers, and validated Q1_0 row mutation. Miyagi owns the patch format,
architecture mapping, probes, fitness policy, search orchestration, reports,
and CLI workflows.

## The Mental Model

Miyagi works with three artifacts:

- **Model**: a GGUF file loaded through wwama. Miyagi currently maps
  `blk.<layer>.ffn_{gate,up,down}.weight` Q1_0 tensors.
- **Probe set**: prompts with a `correct` and `wrong` answer string. Miyagi
  measures the selected-logit gap: `correct_logit - wrong_logit`.
- **Patch**: a JSON list of logical `(layer, projection, row)` coordinates. A
  patch XORs the packed Q1_0 row bytes and can be removed by applying the same
  coordinates again.

The normal workflow is:

```text
inspect model -> define targets and controls -> search or load a patch
-> evaluate target and preservation behavior -> test held-out/generalization
```

A positive target score is only evidence for the probes that were measured. It
does not prove broad knowledge, safety, generalization, or cross-format parity.

## Agent Skills

The [`skills/`](skills/) directory provides reusable Codex workflows for
operating Miyagi. Each skill includes focused instructions, agent metadata, and
supporting references where the workflow needs a schema, checklist, or report
format.

### Core workflows

- [Inspect a model](skills/miyagi-inspect-model/SKILL.md): verify architecture,
  Q1_0 tensor support, dimensions, placement, and model signature.
- [Author probes](skills/miyagi-author-probes/SKILL.md): create auditable target,
  control, and held-out probe files with explicit token semantics.
- [Evaluate a patch](skills/miyagi-evaluate-patch/SKILL.md): validate an artifact
  and compare baseline versus patched probes and generation.
- [Search for a patch](skills/miyagi-search-patch/SKILL.md): run deterministic,
  checkpointed search with explicit targets and preservation controls.
- [Compose patches](skills/miyagi-compose-patches/SKILL.md): combine row-XOR
  patches using symmetric-difference semantics and validate the result.
- [Benchmark a patch](skills/miyagi-benchmark-patch/SKILL.md): measure baseline
  and patched generation on JSON or JSONL datasets.

### Adaptation and validation goals

- [Inject verified knowledge](skills/miyagi-inject-knowledge/SKILL.md): adapt from
  a sourced fact ledger without inventing answers or provenance.
- [Correct known errors](skills/miyagi-correct-errors/SKILL.md): repair observed
  factual or completion errors while testing nearby behavior.
- [Preserve capabilities](skills/miyagi-preserve-capabilities/SKILL.md): define
  and enforce regression gates around a target adaptation.
- [Test generalization](skills/miyagi-test-generalization/SKILL.md): measure
  transfer to paraphrases, held-out facts, contexts, and datasets.
- [Diagnose regressions](skills/miyagi-diagnose-regression/SKILL.md): isolate
  target gains, control losses, patch interactions, and failing gates.
- [Specialize a domain](skills/miyagi-specialize-domain/SKILL.md): design a
  bounded domain adaptation with adjacent-domain and general controls.
- [Suppress unwanted behavior](skills/miyagi-suppress-behavior/SKILL.md): reduce
  a measured response tendency while defining a replacement behavior.
- [Select a patch](skills/miyagi-select-patch/SKILL.md): compare candidates under
  common evidence gates and choose the smallest or safest passing artifact.

## Requirements

- A Rust toolchain compatible with edition 2024.
- The sibling `wwama` crate available at `../wwama`.
- A supported Q1_0 GGUF model for mutation, search, or patch evaluation.
- Enough memory for a writable model session. Mutable tensors disable mmap in
  wwama and can use substantially more memory than read-only inspection.

Use the default CPU build for a first run. Forward native accelerator features
with `--features cuda` or `--features vulkan` when the corresponding wwama
backend is available.

## Quick Start

Set a model path in the examples below:

```sh
MODEL=~/Models/llm/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf
```

### 1. Inspect the model

Inspection is read-only and does not enable mutable tensors. Use CPU placement
for a predictable capability check:

```sh
cargo run --release --no-default-features -- --json inspect \
  --model "$MODEL" \
  --n-gpu-layers 0
```

Check `miyagi_supported`, the architecture signature, layer count, mapped
projection dimensions, row counts, and backend placement. Stop if the report
contains an architecture error.

Use `--all-tensors` when diagnosing a missing mapping or backend placement:

```sh
cargo run --release --no-default-features -- inspect \
  --model "$MODEL" \
  --n-gpu-layers 0 \
  --all-tensors
```

### 2. Inspect a patch

Read a patch without loading a model:

```sh
cargo run --release --no-default-features -- --json info \
  --patch patches/calculus_v1.json
```

Validate coordinates and architecture identity against a live model:

```sh
cargo run --release --no-default-features -- --json info \
  --patch patches/calculus_v1.json \
  --model "$MODEL" \
  --n-gpu-layers 0
```

Use `--allow-model-mismatch` only for an explicitly bounded experiment. It
allows structural validation to continue; it does not establish that a patch
trained for another model or format has equivalent behavior.

### 3. Evaluate probe behavior

Evaluate a patch against built-in math, code, and knowledge probes:

```sh
cargo run --release --no-default-features -- --json eval \
  --model "$MODEL" \
  --n-gpu-layers 0 \
  --patch patches/calculus_v1.json \
  --probes math,code,knowledge \
  --token-mode compatibility \
  --report calculus-evaluation.json
```

`eval` measures baseline and patched gaps in one writable session, removes the
patch before successful exit, and reports per-probe deltas and sign transitions.
Use `--token-mode strict` for new probe sets when every answer must be exactly
one token. Compatibility mode preserves final-token behavior for multi-token
answer strings.

### 4. Compare generation

Compare deterministic generation without and with a patch:

```sh
cargo run --release --no-default-features -- --json apply \
  --model "$MODEL" \
  --n-gpu-layers 0 \
  --patch patches/calculus_v1.json \
  --prompt 'Explain why 7 * 8 equals 56.' \
  --max-tokens 120 \
  --seed 42
```

The command uses the same generation settings for both outputs and restores the
model state before exit. It does not rewrite the GGUF file.

### 5. Search for a patch

Search uses scale-weighted candidate sampling, two-probe screening, deterministic
SplitMix64 sampling, control-penalized fitness, strict-improvement acceptance,
and XOR rollback for rejected candidates.

Use explicit layers rather than assuming default layers fit every model:

```sh
cargo run --release --no-default-features -- search \
  --model "$MODEL" \
  --n-gpu-layers 0 \
  --target math \
  --control code \
  --control knowledge \
  --layers 1,2,3,4 \
  --projections gate_proj,up_proj \
  --iters 200 \
  --fitness mean \
  --penalty 2.0 \
  --seed 42 \
  --screen-probes 2 \
  --checkpoint math.checkpoint.json \
  --report math.search.json \
  --output math.patch.json \
  --name math-adaptation-v1
```

Search writes the patch only after successful completion. Accepted flips remain
applied in the current process until it exits; the model file is never changed.
With `--checkpoint`, cancellation saves resumable state. Resume with the same
model, probes, layers, projections, seed, fitness, penalty, screen count, and
patch metadata; increase only the iteration ceiling:

```sh
cargo run --release --no-default-features -- search \
  --model "$MODEL" \
  --n-gpu-layers 0 \
  --target math \
  --control code \
  --control knowledge \
  --layers 1,2,3,4 \
  --projections gate_proj,up_proj \
  --iters 400 \
  --fitness mean \
  --penalty 2.0 \
  --seed 42 \
  --screen-probes 2 \
  --token-mode compatibility \
  --name math-adaptation-v1 \
  --resume math.checkpoint.json \
  --output math.patch.json
```

### 6. Compose patches

Composition uses XOR symmetric difference: coordinates present in an even number
of inputs cancel, and coordinates present in an odd number remain.

```sh
cargo run --release --no-default-features -- --json compose \
  --patch first.patch.json second.patch.json \
  --name combined-v1 \
  --output combined.patch.json
```

The input patches must have the same `base_model`. Validate the result against
the intended model and evaluate the combined behavior; patch interactions are
not predictable from flip counts alone.

### 7. Benchmark generation

Miyagi accepts a JSON array or JSONL dataset. Each selected record must contain
string question and answer fields. The regexes must put the value to compare in
capture group 1.

```sh
cargo run --release --no-default-features -- --json benchmark \
  --model "$MODEL" \
  --n-gpu-layers 0 \
  --dataset tests/fixtures/smoke_dataset.json \
  --patch math.patch.json \
  --prompt-template "Solve the problem and end with 'The answer is [number]'. {question}" \
  --answer-regex '(?i)the answer is[:\s]*\$?([-\d,]+)' \
  --gold-regex '^([-\d,]+)$' \
  --limit 20 \
  --seed 42 \
  --report math.benchmark.json
```

The report includes baseline and patched per-case responses, extracted answers,
accuracy counts, and `model_restored`. A small smoke dataset is not evidence of
broad generalization or safety.

## Custom Probes

Custom probe files are JSON arrays:

```json
[
  {
    "prompt": "The capital of Australia is",
    "correct": " Canberra",
    "wrong": " Sydney",
    "name": "australia_capital",
    "category": "geography"
  }
]
```

Probe names must be unique. Keep leading whitespace in answer strings when it is
part of the tokenizer behavior. Use separate files for target, control, and
held-out probes. See the [probe authoring skill](skills/miyagi-author-probes/SKILL.md)
for validation guidance.

Built-in selectors are `math`, `code`, and `knowledge`. Any selector that is
not one of those names is treated as a JSON probe-file path.

## Patch Format

Miyagi writes the canonical `miyagi_row_xor_v1` schema:

```json
{
  "version": 1,
  "format": "miyagi_row_xor_v1",
  "name": "example",
  "description": "Sparse row adaptation",
  "base_model": "model.gguf",
  "flips": [
    {"layer": 4, "proj": "gate_proj", "row": 123}
  ],
  "stats": {
    "n_flips": 1,
    "logical_bits_flipped": 4096,
    "compact_binary_estimate_bytes": 12
  },
  "metadata": {}
}
```

The reader also accepts the alternate `type: "row_flip"` field. Patch
validation checks version, format, projection, duplicate coordinates, row
bounds, Q1_0 support, and architecture signature. Logical bit counts come from
the live tensor width, not a hard-coded projection size.

## CLI Reference

| Command | Purpose | Model load |
| --- | --- | --- |
| `inspect` | Discover supported tensors and architecture | Read-only |
| `info` | Read a patch; optionally validate it against a model | Read-only |
| `compose` | XOR-compose two or more patches | None |
| `eval` | Compare baseline and patched probe gaps | Mutable |
| `apply` | Compare baseline and patched generation | Mutable |
| `search` | Create a patch with screened greedy search | Mutable |
| `benchmark` | Score baseline and optional patched generation | Mutable when patched |

Use `--json` for machine-readable reports. Search progress with `--json` is a
stream of JSON events followed by the final result; use `--report` when a clean
JSON artifact is needed.

## Library Surface

The Rust library exposes the same building blocks for application integration:

- `ArchitectureMap`, `Projection`, and `TensorInfo` for live model mapping;
- `WwamaBackend` and `MiyagiBackend` for model operations;
- `Patch`, `ValidatedPatch`, and `PatchFlip` for patch lifecycle management;
- `Probe`, `CompiledProbe`, and `ProbeMeasurement` for probe evaluation;
- `FitnessMode` for mean or minimum target improvement; and
- `SearchConfig`, `SearchCheckpoint`, and `SearchResult` for deterministic
  search orchestration.

## Build and Test

CPU build and tests:

```sh
cargo check --no-default-features
cargo test --no-default-features --all-targets
```

CUDA or Vulkan feature builds:

```sh
cargo check --features cuda
cargo check --features vulkan
```

The large-model integration tests are opt-in and require a local fixture:

```sh
MIYAGI_TEST_MODEL="$MODEL" \
MIYAGI_TEST_GPU_LAYERS=0 \
cargo test --no-default-features --test model_backend
```

### Building on Windows (MSVC)

Miyagi builds and runs on Windows (validated on Windows 11, Rust
`x86_64-pc-windows-msvc`, against mainline llama.cpp b10054 with an RTX 4090).
Notes:

- With Visual Studio 18 installed, the `cmake` crate requests the
  "Visual Studio 18 2026" generator, which needs CMake ≥ 4.3 on `PATH`
  (VS's bundled CMake works).
- For `--features cuda` with a VS version newer than your CUDA toolkit's VS
  integration, use the Ninja generator inside a `vcvars64` environment of a VS
  version the toolkit supports, e.g.:

  ```bat
  call "...\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  set CMAKE_GENERATOR=Ninja
  cargo build --release --features cuda
  ```

## Compatibility and Boundaries

- Miyagi supports descriptor-driven Qwen/Bonsai MLP mapping for two-dimensional
  Q1_0 tensors. Unsupported architectures fail with structured capability
  errors before mutation.
- Mutation is row-level Q1_0 XOR. Arbitrary bit, group, floating-point, and
  ternary mutation are outside the current contract.
- Model mutation is in memory. Miyagi never rewrites the source GGUF file.
- `eval`, `apply`, and patched `benchmark` restore the model before successful
  exit. Search intentionally returns with accepted flips applied until the
  process exits.
- Concurrent mutation and inference on one session is unsupported.
- Native CPU, CUDA, and Vulkan paths have been validated against local fixtures.
  WebAssembly compilation is possible through wwama, but mutable WebAssembly
  runtime behavior remains unsupported without a runtime fixture.
- Structural patch compatibility does not prove behavioral compatibility between
  MLX-trained artifacts and converted GGUF models. Use live evaluation and
  held-out tests.

Miyagi does not modify llama.cpp source files.

Miyagi is inspired by [nikshepsvn/bankai](https://github.com/nikshepsvn/bankai).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
