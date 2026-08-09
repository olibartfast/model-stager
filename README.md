# model-stager

Server-neutral model repository staging and TensorRT preparation for Kubernetes
inference workloads.

Turns a directory of exported model artifacts into a versioned model repository
tree, then gets out of the way:

```
/staging/detector.onnx          →   detector/1/model.plan      (compiled)
/staging/segmenter.xml + .bin   →   segmenter/1/model.xml + .bin
/staging/pose.torchscript       →   pose/1/model.pt
```

`<model>/<version>/<file>` is the layout Triton, OpenVINO Model Server, and
neuriplo-kserve-runtime all read. Nothing here links the tree to the process
that serves it, so one image is an init container in front of any of them.

- [Why a staging step exists](#why-a-staging-step-exists)
- [Quick start](#quick-start)
- [Staging shapes](#staging-shapes)
- [The dispatch](#the-dispatch)
- [Layouts](#layouts)
- [TensorRT](#tensorrt)
- [Configuration](#configuration)
- [Guarantees](#guarantees)
- [Failure modes](#failure-modes)
- [Development](#development)

## Why a staging step exists

Most model formats are portable: export once, ship the file, serve it anywhere.
A TensorRT engine is not. It is compiled against a specific GPU, driver, and
TensorRT version, and an engine built in CI, on a laptop, or in a `docker build`
layer will not load on the cluster node.

So the artifact you ship cannot always be the artifact you serve, and something
has to sit between them. That is this tool. Once it exists, running portable
formats through the same path as a copy is nearly free, which is why there is
one staging procedure rather than two.

## Quick start

As a Kubernetes init container in front of stock Triton:

```yaml
initContainers:
  - name: stage-models
    image: model-stager:trt
    env:
      - {name: REPOSITORY_LAYOUT, value: triton}
    resources: {limits: {nvidia.com/gpu: 1}}
    volumeMounts:
      - {name: staging, mountPath: /staging}
      - {name: models,  mountPath: /models}
containers:
  - name: tritonserver
    image: nvcr.io/nvidia/tritonserver:25.12-py3
    args: [tritonserver, --model-repository=/models/repo]
    volumeMounts:
      - {name: models, mountPath: /models, readOnly: true}
```

Full manifests in [examples/kubernetes](examples/kubernetes); the same flow
without Kubernetes in [examples/compose](examples/compose).

Locally, with no container at all:

```bash
STAGE_DIR=./exports MODEL_REPOSITORY=./repo REPOSITORY_LAYOUT=triton \
  bin/model-stager
```

## Staging shapes

Two are accepted, and a staging directory may mix them.

**Flat** — one file per model. The filename is the model name.

```
/staging/detector.onnx        →  detector/1/model.plan
/staging/ecdet.pte            →  ecdet/1/model.pte
/staging/segmenter.xml        →  segmenter/1/model.xml
/staging/segmenter.bin        →  segmenter/1/model.bin
/staging/ecdet.pbtxt          →  ecdet/config.pbtxt
```

**Tree** — already repository-shaped, copied through without interpretation.

```
/staging/raft/3/model.plan    →  raft/3/model.plan
/staging/raft/3/labels.txt    →  raft/3/labels.txt
/staging/raft/config.pbtxt    →  raft/config.pbtxt
```

Tree form is the escape hatch. A model needing an exact layout, extra files
beside it, or several versions at once expresses that directly, instead of this
tool growing a configuration language to describe the same thing indirectly.

## The dispatch

| Staged | Becomes | Action |
|---|---|---|
| `.onnx` | engine | `trtexec` compile — or copied, with `ONNX_BACKEND=onnx_runtime` |
| `.plan`, `.engine` | engine | copy — a prebuilt engine, valid only if built on this node |
| `.xml` | IR pair | copy `.xml` **and** rename its `.bin` to match |
| `.pte`, `.tflite`, `.torchscript`, `.pt`, `.pb`, `.dali`, `.json` | as-is | copy |
| `.bin` | — | consumed with its `.xml`; an orphan is an error |
| `.pbtxt` | `<model>/config.pbtxt` | copy beside the version directory |
| anything else | — | **error**, unless `STAGER_IGNORE_UNKNOWN=true` |

That last row is deliberate. Silently skipping an unrecognized artifact yields a
repository quietly missing a model, and the failure then surfaces as a 404 from a
client long after the cause is out of sight.

OpenVINO resolves weights by basename, which is why the `.bin` is renamed rather
than copied as-is — a plain copy produces a model that loads without weights.

## Layouts

Servers agree on `<model>/<version>/` and disagree on what the file inside is
called. `REPOSITORY_LAYOUT` picks the convention.

| Staged | `triton` (default) | `ovms` | `neuriplo` |
|---|---|---|---|
| TensorRT engine | `model.plan` | *refused* | `model.plan` |
| ONNX | `model.onnx` | `model.onnx` | `model.onnx` |
| OpenVINO IR | `model.xml` + `.bin` | `model.xml` + `.bin` | `model.xml` + `.bin` |
| TorchScript | `model.pt` | *refused* | `model.torchscript` |
| TF frozen graph | `model.graphdef` | `model.pb` | `model.pb` |
| TFLite | *refused* | *refused* | `model.tflite` |
| ExecuTorch | *refused* | *refused* | `model.pte` |
| DALI | `model.dali` | *refused* | `model.dali` |
| Ensemble graph | *refused* | *refused* | `model.json` |

"Refused" is a hard error naming the layout and the format, raised **before** any
conversion runs. Writing the file anyway would cost minutes of engine build and
produce a model the server never mentions, because a filename it does not
recognize is one it silently ignores.

Triton declares ensembles in `config.pbtxt` with `platform: "ensemble"` rather
than as a graph file, so a `.json` ensemble does not carry across. Stage a Triton
ensemble in tree form.

## TensorRT

### Static models

Leave the shape variables unset. `trtexec` rejects explicit shapes on a model
with fixed input dimensions:

```
Static model does not take explicit shapes
```

### Dynamic models

A model with dynamic axes needs an **optimization profile**, and a profile is
`min`/`opt`/`max` together:

```bash
TRT_MIN_SHAPES=input1:1x3x256x256
TRT_OPT_SHAPES=input1:1x3x520x960
TRT_MAX_SHAPES=input1:1x3x1080x1920
```

`TRT_SHAPES` is **not** a shorthand for these. It sets a single inference shape
and builds no profile, so an engine built with it cannot accept the range it was
supposed to serve. Setting both is an error, as is setting only some of the
three.

### Mixed repositories

One static and one dynamic model in the same repository cannot share a global
shape setting — set `TRT_SHAPES` globally and every static model's build fails.
Append the model name, uppercased with non-alphanumerics replaced by
underscores:

```bash
TRT_MIN_SHAPES_RAFT_LARGE=input1:1x3x256x256
TRT_OPT_SHAPES_RAFT_LARGE=input1:1x3x520x960
TRT_MAX_SHAPES_RAFT_LARGE=input1:1x3x1080x1920
TRT_PRECISION_YOLO26N_DEPTH=fp32
```

`TRT_SHAPES`, `TRT_MIN_SHAPES`, `TRT_OPT_SHAPES`, `TRT_MAX_SHAPES`,
`TRT_PRECISION` and `TRT_EXTRA_ARGS` all take the suffix. A per-model value wins;
the global applies to everything else. No model is named in any manifest
*structure* — only in a variable whose value is the operator's business.

### Cost

Measured on an RTX 3060: **~8.5 min for a 101 MB model, ~2 min for a 21 MB one.**
Mount `MODEL_REPOSITORY` on a persistent volume rather than an `emptyDir`, or
that time is paid on every pod replacement. Staging is idempotent, so a restart
against a warm volume finishes in seconds.

Budget for the cold case in Kubernetes: `progressDeadlineSeconds` must cover the
build, or the rollout is marked failed while it is still legitimately running.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `STAGE_DIR` | `/staging` | where staged artifacts are read from |
| `MODEL_REPOSITORY` | `/models/repo` | repository root to build |
| `MODEL_VERSION` | `1` | version directory for flat-form artifacts |
| `REPOSITORY_LAYOUT` | `triton` | `triton`, `ovms`, or `neuriplo` |
| `SERVER_EXEC` | *unset* | when set, `exec` this after staging (see below) |
| `ONNX_BACKEND` | `tensorrt` | `tensorrt` (compile) or `onnx_runtime` (copy) |
| `TRT_PRECISION` | `fp16` | `fp16`, `fp32`, or `best` |
| `TRT_SHAPES` | *unset* | single shape spec; **not** a profile |
| `TRT_MIN_SHAPES` / `TRT_OPT_SHAPES` / `TRT_MAX_SHAPES` | *unset* | optimization profile; all three required together |
| `TRT_EXTRA_ARGS` | *unset* | extra `trtexec` arguments, word-split |
| `TRT_FALLBACK_ONNX` | `false` | keep the ONNX if `trtexec` is missing |
| `STAGER_IGNORE_UNKNOWN` | `false` | skip unrecognized files instead of failing |

### `SERVER_EXEC`

By default the stager stages and exits — an init container should not start
anything. Setting `SERVER_EXEC` makes it `exec` a server afterwards, with the
container's own arguments appended:

```bash
SERVER_EXEC="tritonserver --model-repository=/models/repo"
```

This is only needed when the build must happen *inside* the serving container,
which is the narrower case. Prefer the default.

## Guarantees

These hold on every run, because staging re-runs on every restart:

1. **Idempotent.** A model whose output already exists is skipped. A restart
   against a warm volume rebuilds nothing.
2. **Atomic.** Each version is built in a temp directory and published by
   renaming that directory. A per-file rename would be enough for a single-file
   model but not for OpenVINO's `.xml` + `.bin`, where a crash between two
   renames leaves a model that loads without weights.
3. **No silent substitution.** If the requested backend cannot be produced, the
   run fails. Serving a different backend has a different latency and accuracy
   profile, so it is only ever opt-in (`TRT_FALLBACK_ONNX=true`).
4. **Nothing is named.** The model name comes from the filename, the backend
   from the filename written. Adding or removing a model changes no manifest.
5. **`exec` last.** With `SERVER_EXEC`, the server replaces this process, so
   signals and exit codes propagate.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Static model does not take explicit shapes` | shape variables set for a fixed-dimension ONNX | unset them; they are for dynamic axes |
| Engine rejects the shapes it should serve | built with `TRT_SHAPES` instead of a profile | use `TRT_MIN`/`OPT`/`MAX_SHAPES` |
| `needs TRT_MIN_SHAPES, TRT_OPT_SHAPES and TRT_MAX_SHAPES together` | partial profile | set all three |
| `no servable artifacts staged in /staging` | nothing staged, or the volume is not shared | check both containers mount the same staging volume |
| `unrecognized staged file` | an artifact with no dispatch arm | add one, or `STAGER_IGNORE_UNKNOWN=true` |
| `... has no matching .xml` | OpenVINO weights staged without the model | stage both, same basename |
| `layout '<x>' cannot serve a .<ext> model` | format the target server cannot load | change layout, or export a format it can serve |
| `trtexec not found in PATH` | image has no TensorRT | use the TensorRT base, or `ONNX_BACKEND=onnx_runtime` |
| Version mismatch when the server loads an engine | stager and server ship different TensorRT | align the image tags |
| Rollout failed while logs show a build running | `progressDeadlineSeconds` shorter than the build | raise it |

## Development

```bash
tests/test-model-stager.sh
```

97 assertions, no GPU, no TensorRT, no server: `trtexec` and the server are
stubs on `PATH` that record how they were called. What is under test is the
dispatch and the tree it produces.

`bin/model-stager` is POSIX `sh` and stays that way — the container shell is
often busybox `ash`. CI runs the suite under `bash` and `dash`, and lints with
`shellcheck`.

It is also deliberately a single self-contained file, so it can be `COPY`d into
any image without carrying a directory of sourced helpers around with it.

## License

MIT. See [LICENSE](LICENSE).
