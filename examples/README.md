# Examples

The examples keep model selection, storage, and runtime selection as deployment
inputs. No manifest encodes a task, tensor, shape, or built-in model identity.

| File | Purpose |
|---|---|
| [kubernetes/stager-job.yaml](kubernetes/stager-job.yaml) | Stage a selected catalog from one PVC into a repository PVC. |
| [kubernetes/kserve.yaml](kubernetes/kserve.yaml) | Let [KServe](https://github.com/kserve/kserve) mount an already runtime-shaped repository through `modelFormat` and `storageUri`. |
| [kubernetes/triton.yaml](kubernetes/triton.yaml) | Run the stager as an init container before [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server). |
| [kubernetes/ovms.yaml](kubernetes/ovms.yaml) | Run the same neutral stager before [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server). |
| [compose/docker-compose.yml](compose/docker-compose.yml) | The Triton init flow expressed with Compose. |

## Choose the flow by input shape

An already runtime-shaped repository needs no transformation. KServe's storage
initializer copies or mounts the `storageUri` contents at `/mnt/models`, and the
selected runtime reads them there.

Raw or mixed artifacts first need a local staging step:

```text
input storage -> /staging -> model-stager -> /model_repository
```

The input list is authoritative:

```yaml
- {name: MODELS, value: "model-a,model-b"}
```

For independent paths rather than one catalog:

```yaml
- name: MODEL_INPUTS
  value: |
    model-a=/mnt/input-a/model.onnx
    model-b=/mnt/input-b/model-b
```

The neutral image preserves ONNX and requests no GPU. A deployment that
explicitly wants TensorRT conversion sets `ONNX_BACKEND=tensorrt`, uses an image
containing `trtexec`, and supplies the GPU and profile configuration required by
that conversion. That is an optional capability, not the stager's identity.

Only one stager may write a repository at a time. The Deployment examples use
`strategy.type: Recreate` so a rollout cannot overlap two init containers on
the same repository PVC; staging Jobs must likewise not overlap.

`/model_repository` is the generic image default. `/mnt/models` is KServe's
default local model mount. Set `MODEL_REPOSITORY` explicitly whenever the stager
writes into a KServe-owned volume.
