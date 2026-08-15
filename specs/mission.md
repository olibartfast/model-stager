# Mission

`model-stager` maps local model artifacts into a versioned model repository tree
and exits.

That is the whole product. Everything below exists to keep it that small.

## The problem

A serving runtime does not load a directory of exported files. It loads a
repository tree — `<model>/<version>/<file>` — where the *filename* declares the
backend. The layout was introduced by
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
and is also read by
[OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) for the
formats it supports.

Between "a training pipeline produced `detector.onnx`" and "a runtime serves
it", something has to rename, version, and place that file. In practice that
something is an ad-hoc shell snippet in an init container, rewritten per project,
per runtime, per model. Those snippets share three failure modes:

1. **They know their model.** The model's name, tensor names, or task are baked
   into the script, so the script is not reusable and the deployment manifest
   changes whenever the served model changes.
2. **They fail quietly.** An unrecognized extension is skipped, a filename the
   runtime does not recognize is written, and the model is simply never loaded.
   The symptom arrives much later as a 404 from a client, far from its cause.
3. **They are not restart-safe.** Staging re-runs on every pod restart. A script
   that copies file-by-file can be observed mid-copy by a runtime that is already
   watching the directory.

## What this project is

One POSIX `sh` file — `bin/model-stager` — that solves exactly that mapping.

It reads local artifacts, writes a repository tree, and exits with a status. It
is the init container in front of a serving runtime, and it is neutral about
which runtime that is.

## What this project is not

Naming these is more useful than naming the features, because every one of them
is a thing operators reasonably expect and will not get:

- **Not a downloader.** No S3, GCS, PVC, HTTP, registry, or Hugging Face client.
  [KServe](https://github.com/kserve/kserve)'s storage initializer, a volume, or
  the orchestrator makes artifacts local before this tool runs.
- **Not a runtime selector.** It writes a filename convention. It does not know a
  runtime's protocol, flags, or lifecycle.
- **Not a model inspector.** It never reads a tensor, a shape, or a task label.
  `config.pbtxt` is carried through opaque, with one exception: an optional
  top-level `name:` is checked for consistency with the destination directory.
- **Not a conversion framework.** One conversion exists — ONNX to TensorRT via
  `trtexec` — and only because a TensorRT engine is specific to the GPU, driver,
  and TensorRT version that built it and therefore cannot be built at image build
  time. It is an explicit operator request, never inferred.
- **Not a config language.** Tree-form input is the escape hatch. A model needing
  an exact layout expresses it as a directory rather than provoking this script
  into growing a DSL.

## Who this is for

The operator standing up a serving deployment who has artifacts on a volume and
needs a repository tree, and who wants the manifest around the workload to stay
identical whatever is being served. `MODELS` is the single place a deployment
names a model; nothing else in the pipeline has to change.

Secondary: this repository is the author's own, extracted from
[neuriplo-kserve-runtime](https://github.com/olibartfast/neuriplo-kserve-runtime)
— also the author's, an experimental runtime with no user base — where it had
been a TensorRT-to-`neuriplo` adapter before being generalized. The `neuriplo`
layout exists for that runtime and should be ignored by anyone not using it.

## What success looks like

- An operator can point the tool at a volume and get a tree a runtime loads,
  without reading the source.
- A staging failure names the artifact and the reason on the first line of
  stderr.
- A restart that changes nothing does nothing, and says so.
- A restart that changes something either publishes atomically or refuses.
- Adding a format is four mechanical edits (see `AGENTS.md`), not a redesign.

## The philosophy, in one line

A dropped model is worse than a failed pod. Refuse loudly; never substitute.
