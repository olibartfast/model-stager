# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Made ONNX pass-through and the BusyBox image backend-neutral defaults;
  TensorRT conversion is now explicit.
- Added `MODEL_INPUTS` for newline-separated `name=local-path` inputs alongside
  catalog selection through `STAGE_DIR` and `MODELS`.
- Recognize complete repositories and mounted single-model trees, validate
  repositories in place, and refuse ambiguous directories or in-place filters.
- Use `/model_repository` as the generic output default and document
  [KServe](https://github.com/kserve/kserve)'s `/mnt/models` path separately.
- Replaced model- and task-specific examples with neutral direct, staging-job,
  and KServe `modelFormat`/`storageUri` flows.
- Validate numeric flat-form versions, duplicate inputs, config selection, and
  layout-specific
  [Triton](https://github.com/triton-inference-server/server) ensembles before
  publication.
- Verify warm versions against request, source, and output fingerprints instead
  of treating directory existence as sufficient. Existing 0.1.0 versions
  without metadata are adopted when their requested output is directly
  verifiable; metadata-free TensorRT engines require an explicit, warning-emitting
  `STAGER_ADOPT_UNVERIFIED_TENSORRT=true` trust decision.
- Keep `config.pbtxt` outside version identity and publish config changes
  atomically without forcing model or engine rebuilds.
- Refuse a non-empty existing ensemble version, keep only output-affecting
  properties in flat-version identity, and clean interrupted metadata writes.
- Apply `STAGER_IGNORE_UNKNOWN` to hidden repository-shaped model-root entries,
  with precise diagnostics when the escape hatch is disabled.
- Refuse normalized model-name collisions only when an active per-model
  TensorRT override would otherwise be shared silently.
- Report config overlay application only when the config was actually written.
- Exercise config parsing inside the default BusyBox image in CI.
- Refuse config-name mismatches, validate portable model names, isolate
  conversion-tool stdin, and support TensorFlow Lite under the `ovms` layout.
- Expanded `tests/test-model-stager.sh` to 271 assertions.

## [0.1.0] - 2026-08-09

### Added

- `bin/model-stager`: builds a versioned model repository tree from staged
  artifacts and exits. Extracted from
  [neuriplo-kserve-runtime](https://github.com/olibartfast/neuriplo-kserve-runtime),
  the author's own experimental runtime, where it had been a
  TensorRT-to-neuriplo adapter, and generalized.
- Flat and tree staging shapes, mixable in one staging directory.
- Dispatch across ONNX, TensorRT, OpenVINO IR, TorchScript, TensorFlow, TFLite,
  ExecuTorch, DALI, and ensemble graphs.
- `REPOSITORY_LAYOUT` for `triton`, `ovms`
  ([OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)), and
  `neuriplo` filename conventions, with formats the target server cannot load
  refused before any conversion runs.
  `neuriplo` targets this author's own runtime, not a third-party server.
- TensorRT optimization profiles via `TRT_MIN_SHAPES` / `TRT_OPT_SHAPES` /
  `TRT_MAX_SHAPES`. A single `TRT_SHAPES` builds no profile, so it could not
  correctly build a dynamic-axis engine.
- Per-model overrides for every TensorRT variable, so one static and one dynamic
  model can share a repository.
- `SERVER_EXEC` to optionally `exec` a server after staging. Unset by default:
  an init container should not start anything.
- `Dockerfile` with a swappable base — TensorRT for engine builds, `busybox` for
  a repository that needs no compilation.
- Kubernetes examples for Triton and OVMS, and a compose example.
- `tests/test-model-stager.sh`: 132 assertions with `trtexec` and the server
  stubbed on `PATH`, so no GPU, TensorRT, or server is required. CI runs it
  under `bash` and `dash`.
