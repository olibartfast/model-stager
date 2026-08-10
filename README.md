# model-stager

`model-stager` maps local model artifacts into a versioned repository and exits.
It is model-agnostic, task-agnostic, and neutral about the process that serves
the result.

The `<model>/<version>/<file>` tree was introduced by
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).
[OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) also
reads compatible trees for the formats it supports.

`bin/model-stager` is the whole tool. It does not download remote URIs, select a
serving runtime, inspect tensors, or interpret task metadata.

## Input contracts

The stager accepts local inputs in two mutually exclusive modes.

### Catalog mode: `STAGE_DIR` plus `MODELS`

`STAGE_DIR` may mix flat artifacts and repository-shaped model directories:

```text
/staging/model-a.onnx
/staging/model-a.pbtxt
/staging/model-b.xml
/staging/model-b.bin
/staging/model-c/3/model.plan
/staging/model-c/config.pbtxt
```

`MODELS=model-a,model-c` stages exactly that subset. An unselected artifact or
config is ignored; a requested name that is absent is an error. With `MODELS`
unset, every recognized model is staged. Names use letters, digits, dots,
underscores, and hyphens; hidden catalog entries are errors unless
`STAGER_IGNORE_UNKNOWN=true`.

Selection is literal, not dependency resolution. In particular, selecting a
config-only ensemble does not automatically select the models referenced by its
graph. The operator's list must include the complete serving set.

A complete existing repository can be mounted as `STAGE_DIR`. Each child model
directory is selected and copied verbatim. When source and destination are the
same directory, an unfiltered run validates the repository in place; filtering
in place is refused because the unselected directories would remain visible.

A mounted single-model directory has numeric version directories directly at
its root. Its name is supplied through the one-entry list:

```bash
STAGE_DIR=/input/model \
MODELS=model-a \
MODEL_REPOSITORY=/model_repository \
bin/model-stager
```

### Explicit mode: `MODEL_INPUTS`

Independent local inputs use newline-separated `name=path` records:

```bash
MODEL_INPUTS='model-a=/mnt/input/model-a.onnx
model-b=/mnt/repository/model-b' \
REPOSITORY_LAYOUT=triton \
MODEL_REPOSITORY=/model_repository \
bin/model-stager
```

Each path must be either one recognized flat artifact or one repository-shaped
model directory. A flat artifact also carries a same-basename sibling
`config.pbtxt` when present. Names must be unique.

`MODEL_INPUTS` and `MODELS` cannot be set together. This is deliberate:
explicit paths and catalog selection are separate, auditable input contracts;
the stager does not merge an implicit catalog with explicit records.

Directory recognition is structural: numeric version directories plus an
optional `config.pbtxt`. Unexpected hidden entries at that model root are
refused by default and ignored when `STAGER_IGNORE_UNKNOWN=true`; files inside
numeric version directories, including dotfiles, are copied verbatim. Ambiguous
directories fail rather than being guessed
as a backend-specific model.

The config remains opaque except for an optional top-level `name:` consistency
check. If present, that name must match the destination directory; the stager
never rewrites it. Configs belong to the model directory rather than a numbered
version and may be updated independently; each update is published atomically.

## Output layouts

`REPOSITORY_LAYOUT` selects only the filename convention at the destination:

Dispatch is keyed by the input extension:

| Input extension | `triton` (default) | `ovms` | `neuriplo` [^1] |
|---|---|---|---|
| `.plan`, `.engine` | `model.plan` | refused | `model.plan` |
| `.onnx` | `model.onnx` | `model.onnx` | `model.onnx` |
| `.xml` + `.bin` | `model.xml` + `model.bin` | `model.xml` + `model.bin` | `model.xml` + `model.bin` |
| `.torchscript` | `model.pt` | refused | `model.torchscript` |
| `.pt` | `model.pt` | refused | `model.pt` |
| `.pb` | `model.graphdef` | `model.pb` | `model.pb` |
| `.tflite` | refused | `model.tflite` [^2] | `model.tflite` |
| `.pte` | refused | refused | `model.pte` |
| `.dali` | `model.dali` | refused | `model.dali` |
| `.json` | refused | refused | `model.json` |

[^1]: `neuriplo` targets the author's own runtime. Ignore it unless using that
runtime.

[^2]: [OpenVINO Model Server's model-repository documentation](https://docs.openvino.ai/2023.3/ovms_docs_models_repository.html)
lists TensorFlow Lite as a directly loadable format.

A refused combination fails before conversion or publication. Tree-form inputs
are already repository-shaped and are copied verbatim; the operator is
responsible for choosing a tree compatible with the selected runtime.

[Triton ensembles](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)
are config-only models. A flat `config.pbtxt` declaring `platform: "ensemble"`
creates the required empty version directory only for the `triton` layout.

## Backend conversion

The neutral default preserves ONNX:

```text
ONNX_BACKEND=onnx_runtime   # default: copy model.onnx
```

TensorRT conversion is explicit:

```text
ONNX_BACKEND=tensorrt
```

That mode requires `trtexec` in the image and, for a real build, a compatible
GPU. A conversion-capable image can be built with:

```bash
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/tensorrt:25.12-py3 \
  -t model-stager:tensorrt .
```

The TensorRT version used to build an engine must match the serving runtime.
Static models need no shape variables. Dynamic models require all three profile
bounds:

```bash
TRT_MIN_SHAPES=input:1x3x256x256
TRT_OPT_SHAPES=input:1x3x512x512
TRT_MAX_SHAPES=input:1x3x1024x1024
```

Per-model overrides append the normalized model name, for example
`TRT_PRECISION_MODEL_A=fp32`. The stager derives that suffix mechanically; it
contains no built-in model identity. Names such as `model-a` and `model.a` may
coexist unless a non-empty per-model TensorRT override for their shared
`MODEL_A` suffix is active. In that case the stager refuses both rather than
silently building two engines with one model's settings.

## KServe integration

[KServe](https://github.com/kserve/kserve) uses `modelFormat` to select a
compatible serving runtime and `storageUri` or `storageUris` to locate model
artifacts. Its storage initializer makes those artifacts local to the serving
container, normally under `/mnt/models`.

Those fields remain KServe control-plane inputs. `model-stager` neither parses
an `InferenceService` nor downloads S3, GCS, PVC, HTTP, or Hugging Face URIs.

There are two valid flows:

```text
Already runtime-shaped:
storageUri -> KServe storage initializer -> /mnt/models -> runtime
                                            (no stager)

Raw or mixed local artifacts:
storage initializer or volume -> /mnt/input-models
                              -> model-stager
                              -> /mnt/models
                              -> runtime
```

`/mnt/models` is KServe's default local mount. `/model_repository` is this
project's generic container default. Both are supported by setting
`MODEL_REPOSITORY`; neither path changes dispatch behavior.

See [examples](examples/README.md) for direct, staged, and KServe-native flows.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `STAGE_DIR` | `/staging` | catalog root containing flat artifacts or model directories |
| `MODEL_INPUTS` | unset | newline-separated `name=local-path` records; exclusive with `MODELS` |
| `MODELS` | unset | comma- or space-separated catalog selection; unset selects all |
| `MODEL_REPOSITORY` | `/model_repository` | output repository root |
| `MODEL_VERSION` | `1` | numeric version for flat artifacts |
| `REPOSITORY_LAYOUT` | `triton` | `triton`, `ovms`, or `neuriplo` filename convention |
| `ONNX_BACKEND` | `onnx_runtime` | preserve ONNX or explicitly request `tensorrt` conversion |
| `SERVER_EXEC` | unset | optional command to `exec` after successful staging |
| `TRT_PRECISION` | `fp16` | `fp16`, `fp32`, or `best` for explicit TensorRT conversion |
| `TRT_SHAPES` | unset | one fixed inference shape; not an optimization profile |
| `TRT_MIN_SHAPES` / `TRT_OPT_SHAPES` / `TRT_MAX_SHAPES` | unset | dynamic profile; all three required |
| `TRT_EXTRA_ARGS` | unset | extra `trtexec` arguments, word-split |
| `TRT_FALLBACK_ONNX` | `false` | explicit fallback when TensorRT tooling is absent |
| `STAGER_ADOPT_UNVERIFIED_TENSORRT` | `false` | explicitly trust a metadata-free existing TensorRT engine during migration |
| `STAGER_IGNORE_UNKNOWN` | `false` | skip unknown catalog files and hidden model-root entries when true |

## Guarantees

1. **Model- and task-agnostic.** Names come from input records or filenames;
   configs are carried through without interpreting tensors, backends, or tasks.
2. **Backend-neutral by default.** Portable artifacts remain portable unless a
   conversion is explicitly requested.
3. **Fail loud.** Missing, ambiguous, duplicate, unsafe, or unsupported inputs
   are errors.
4. **Verified idempotency.** Flat versions record request and output
   fingerprints; tree versions are compared recursively. A changed source,
   output-affecting backend or build setting, or published output is refused
   until the operator explicitly removes the old version. Equivalent input
   extensions or layouts that resolve to the same target are accepted. Config
   updates are independent and atomic.
5. **Atomic publication.** Version directories and config files are prepared at
   temporary paths and renamed into place.
6. **POSIX `sh`, one file.** The script runs under BusyBox `ash` and `dash` and
   has no sourced helper directory.
7. **Single writer.** Concurrent stagers must not write the same repository.
   Temporary cleanup and publication assume one owning process; deployments
   should use one staging Job or a `Recreate` rollout strategy.

Flat versions contain `.model-stager-state` and `.model-stager-output`.
These dotfiles are deliberate stager metadata inside the atomically published
version directory. A compatible version created by 0.1.0 has neither file; the
first restart adopts it and writes the metadata. Direct-copy artifacts are
compared before adoption. A previously converted TensorRT engine cannot be
verified against its ONNX source, so metadata-free TensorRT versions are
refused by default. Set `STAGER_ADOPT_UNVERIFIED_TENSORRT=true` for a one-time,
warning-emitting adoption only when the operator explicitly trusts the existing
engine.

Verification reads source and published files to compute checksums, while
repository-shaped trees are recursively compared. This favors detecting drift
over minimal restart I/O and can be significant for multi-gigabyte models on a
remote PVC.

## Development

```bash
tests/test-model-stager.sh
shellcheck bin/model-stager tests/test-model-stager.sh
```

The suite currently contains 271 assertions. It stubs `trtexec` and the optional
server command, so it needs no GPU, TensorRT installation, or serving runtime.
CI runs it under both the normal shell and `dash`.
