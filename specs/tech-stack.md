# Tech Stack and Architecture of Record

This document describes `model-stager` as it is built today, not as it is
wished. It is the reference an implementation plan is checked against.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | POSIX `sh` | The container shell is often BusyBox `ash`. No bashisms: no arrays, no `local`, no `[[ ]]`, no `${var,,}`. |
| Unit | One file, `bin/model-stager` | It is `COPY`d into arbitrary images. A directory of sourced helpers would make the tool a package with an install step. |
| External tools | `cksum`, `awk`, `grep`, `cmp`, `find`, `cp`, `mv`, `ls` | All POSIX-specified and all present in BusyBox. |
| Optional tool | `trtexec` | Only for `ONNX_BACKEND=tensorrt`. Absent by default. |
| Image | `Dockerfile` with swappable `BASE_IMAGE` | `busybox` for pass-through, `nvcr.io/nvidia/tensorrt:*` for engine builds. |
| Tests | `tests/test-model-stager.sh`, 271 assertions | Stubs `trtexec` and the server command on `PATH`. No GPU, no TensorRT, no runtime required. |
| Lint | `shellcheck` | Runs over the tool and the suite. |
| CI | `.github/workflows/ci.yml` | Suite under the normal shell and under `dash`; config parsing exercised inside the BusyBox image. |

**No language runtime is permitted.** Python, Go, or a compiled binary would each
solve the readability problem and each break the deployment property that makes
the tool usable: a single file that runs in whatever image already exists.

## Runtime shape

A one-shot process. It stages and exits `0`. It is an init container.

```
volume / storage initializer
        │
        ▼
   local artifacts ──► model-stager ──► <model>/<version>/<file> ──► runtime
                       (exits 0)              (a volume)
```

`SERVER_EXEC` is the single deviation: when set, the tool `exec`s a command after
staging instead of exiting. That exists only because a TensorRT engine must be
built on the machine that serves it, which forces the build into the serving
container rather than an init container.

## Internal layers

The file has an implicit layered structure. It is not marked in the source, and
that is a finding, not a description — see `specs/roadmap.md` Phase 1.

```
L6  Orchestration      main loop, stage_flat, stage_tree,
                       stage_ensemble_config, stage_explicit_inputs
                              │
L5  Publication        publish, copy_config              ← atomicity lives here
                              │
L4  Identity           fingerprint_file, write_flat_metadata, already_staged
                              │
L3  Transformation     convert_onnx                      ← the only conversion
                              │
L2  Dispatch           target_filename, resolve_target,  ← the ONLY layer with
                       is_ensemble_config,                 runtime-facing
                       config_declared_name,               knowledge
                       validate_config_name
                              │
L1  Admission          validate_model_name, model_selected, model_mark_found,
                       validate_explicit_inputs, check_catalog_conflicts,
                       check_hidden_catalog_entries, check_hidden_model_entries,
                       is_single_model_tree, same_directory
                              │
L0  Configuration      environment defaults, model_override,
                       model_override_key, tensorrt_override_key_in_use
```

**The layering is a real constraint, not documentation.** Server-facing knowledge
is confined to L2 — that confinement is what makes the "server-neutral" invariant
checkable rather than aspirational. A change that teaches L6 about a runtime is a
design change.

### The ordering guarantee

Within a single model, the phases must run in this order. Every one of these
orderings was chosen to prevent a specific failure, and reordering them is a
correctness bug even when the code still passes:

1. **Admit** — name validated, selection applied, shape recognized.
2. **Resolve** — `resolve_target` computes the destination filename and refuses
   an unsupported layout/format pair *before any work is done*, so a refusal
   cannot leave a temp directory behind.
3. **Fingerprint** — source content is checksummed and compared against the
   published version's recorded state.
4. **Transform** — conversion runs, into a temp path only.
5. **Publish** — a fully-built temp directory is `mv`'d into place. Never
   file-by-file: an OpenVINO model is `model.xml` *and* `model.bin`, and a
   runtime watching the directory must never see one without the other.
6. **Record** — `.model-stager-output` is written, then `.model-stager-state`.
   State moves last, because its presence is the marker that both landed; an
   interrupted write is therefore retryable rather than a corrupt adoption.

## Data model

Three persistent artifacts, all in the destination repository:

| Path | Owner | Identity |
|---|---|---|
| `<model>/<version>/<file>` | a version | Immutable. Republishing is refused. |
| `<model>/<version>/.model-stager-state` | the stager | Request + source fingerprint of the run that published this version. |
| `<model>/<version>/.model-stager-output` | the stager | Fingerprint of what was actually written. |
| `<model>/config.pbtxt` | the model | *Outside* version identity. Updated atomically and independently, without rebuilding an engine. |

Putting `config.pbtxt` outside version identity is the load-bearing decision in
the data model: a config edit is the common operational change, and coupling it
to version identity would force a multi-gigabyte TensorRT rebuild for a batch-size
tweak.

Transient names, all swept at startup:

- `.stager-tmp.$$` — in-flight version directory
- `.stager-config.$$` — in-flight config
- `.model-stager-state.$$` / `.model-stager-output.$$` — in-flight metadata

## Input contracts

Two, mutually exclusive by design. They are not merged, because an implicit
catalog silently unioned with explicit records is unauditable.

- **Catalog** — `STAGE_DIR` + optional `MODELS`. Scans a directory. Unset
  `MODELS` selects everything; a requested name that is absent is an error.
- **Explicit** — `MODEL_INPUTS`, newline-separated `name=path` records. Each path
  is one recognized flat artifact or one repository-shaped model directory.

Recognition of a directory is *structural*: numeric version directories plus an
optional `config.pbtxt`. Ambiguous directories fail rather than being guessed.

## Configuration surface

18 environment variables, plus per-model TensorRT overrides derived mechanically
by uppercasing the model name and replacing non-alphanumerics with underscores
(`TRT_PRECISION_MODEL_A`). The derivation contains no model identity. Full table:
`README.md`.

`README.md` is the user-facing specification of this surface. `specs/` is the
constitution behind it. They must agree; where they disagree, `README.md`
describes shipped behavior and `specs/` describes intent.

## Known operational weaknesses

These are stated here because the roadmap targets them, and because an operator
deploying this today should know them.

1. **The single-writer contract is documented, not enforced.** Startup sweeps
   `.stager-tmp.*` under the repository unconditionally. Two stagers against one
   repository means one deletes the other's in-flight directory. Nothing detects
   this.
2. **Verification cost scales with model size, every restart.** `fingerprint_file`
   reads the whole source; tree inputs are compared recursively. On multi-gigabyte
   models on a remote PVC this is paid on every pod restart, deliberately trading
   I/O for drift detection — with no way to choose the other trade.
3. **Every failure exits `1`.** An orchestrator cannot distinguish "the config is
   wrong, never retry" from "the volume is not mounted yet, retry".
4. **Output is unstructured `echo`.** There is no run summary and no
   machine-readable record of what was staged.
5. **There is no way to ask without writing.** No dry-run.
6. **Variable hygiene is by convention only.** POSIX `sh` has no `local`, so every
   variable is global. Most functions prefix (`publish_*`, `config_*`,
   `convert_*`), but `stage_flat` uses bare `source_file`, `model_name`,
   `extension`, `target_name` while calling into functions that could shadow them.
   This is currently correct and structurally fragile.

## Open questions

1. Should a repository lock be a lock file, a lease with a TTL, or a
   `mkdir`-based mutex? A crashed stager must not deadlock a restart.
2. Is size-plus-mtime an acceptable fast path for drift detection on a PVC, or
   does it defeat the purpose of the guarantee it replaces?
3. Should the staging manifest be a file in the repository, stdout, or both? A
   file is another thing a runtime might trip over; stdout is lost on restart.
4. Do exit codes belong to categories (input/environment/conflict) or to
   retryability? The orchestrator wants the latter; the operator wants the former.
