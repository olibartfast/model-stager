# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-09

### Added

- `bin/model-stager`: builds a versioned model repository tree from staged
  artifacts and exits. Extracted from `neuriplo-kserve-runtime`, where it had
  been a TensorRT-to-neuriplo adapter, and generalized.
- Flat and tree staging shapes, mixable in one staging directory.
- Dispatch across ONNX, TensorRT, OpenVINO IR, TorchScript, TensorFlow, TFLite,
  ExecuTorch, DALI, and ensemble graphs.
- `REPOSITORY_LAYOUT` for `triton`, `ovms`, and `neuriplo` filename conventions,
  with formats the target server cannot load refused before any conversion runs.
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
