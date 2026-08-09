# Examples

| File | Scenario |
|---|---|
| [kubernetes/triton.yaml](kubernetes/triton.yaml) | One server. Stock Triton served from a staged repository. |
| [kubernetes/ovms.yaml](kubernetes/ovms.yaml) | One server, no GPU. OpenVINO Model Server; staging is copy-and-rename only. |
| [kubernetes/two-servers-explicit.yaml](kubernetes/two-servers-explicit.yaml) | Two servers on one node, client-driven loading. |
| [kubernetes/gpu-time-slicing.yaml](kubernetes/gpu-time-slicing.yaml) | Device plugin config the two-server case needs first. |
| [compose/docker-compose.yml](compose/docker-compose.yml) | The Triton flow without Kubernetes. |

## Two servers, explicit loading

The interesting one. Triton serves segmentation, neuriplo-kserve-runtime serves
pose, both start empty, and the client decides what is resident:

```
                    my-models:seg-v1              my-models:pose-v1
                            │                             │
                            v  /staging                   v  /staging
                    ┌───────────────┐             ┌───────────────┐
                    │ model-stager  │             │ model-stager  │
                    │ LAYOUT=triton │             │ LAYOUT=neuriplo
                    └───────┬───────┘             └───────┬───────┘
                            v                             v
              rfdetr-seg/1/model.plan        rfdetr-pose/1/model.plan
              yolo26-seg/1/model.plan        yolo-pose/1/model.plan
                            │                             │
                            v                             v
                    ┌───────────────┐             ┌───────────────┐
                    │ tritonserver  │             │   neuriplo    │
                    │  --explicit   │             │  --explicit   │
                    └───────────────┘             └───────────────┘
```

One stager image builds both repositories. Only `REPOSITORY_LAYOUT` and the
artifact image differ.

### Two things that will bite before an inference succeeds

**A single-GPU node cannot schedule two GPU pods.** The node advertises
`nvidia.com/gpu: 1`, the first pod takes it, and the second stays `Pending` with
`Insufficient nvidia.com/gpu` — no crash, no failure, just a pod that never
starts. Apply [gpu-time-slicing.yaml](kubernetes/gpu-time-slicing.yaml) first.

Time-slicing shares the device; it does not partition VRAM. Two servers each
holding a large engine can still hit CUDA OOM on a 6 GB card, which is a further
argument for explicit mode: only what the client asked for is resident.

**Explicit mode deadlocks a `/v2/health/ready` readiness probe.**
`neuriplo-kserve-runtime` returns 503 from `/v2/health/ready` while no model is
loaded — measured:

```
$ neuriplo-kserve-runtime --models <repo> --model-control-mode explicit
/v2/health/live   -> HTTP 200
/v2/health/ready  -> HTTP 503
```

With no model loaded that is the normal startup state, so the probe never
passes, the Service gets no endpoints, and nothing can reach the server to issue
the load that would clear it. The process is healthy and unreachable.

Both servers therefore gate readiness on `/v2/health/live` — "the control plane
is up and accepting load requests", which is the only useful meaning of *ready*
for a server whose contents are the client's choice.

Triton with default `--strict-readiness` reports ready once the server is up and
every *loaded* model is ready, vacuously true at zero models, so its probe stays
on `/v2/health/ready`. That is Triton's documented behaviour rather than
something measured here.

### Deploying

```bash
kubectl apply -f kubernetes/gpu-time-slicing.yaml
helm upgrade nvdp nvdp/nvidia-device-plugin -n nvidia-device-plugin \
  --reuse-values --set config.name=nvidia-plugin-config

# Wait for the node to advertise the replicas before going further.
kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'

kubectl apply -f kubernetes/two-servers-explicit.yaml
```

First rollout builds four TensorRT engines and takes minutes. Watch the stagers
rather than guessing:

```bash
kubectl -n neuriplo logs -f deploy/tritonserver -c model-stager
kubectl -n neuriplo logs -f deploy/neuriplo-kserve-runtime -c model-stager
```

### Driving it from the client

Both servers speak the same model-repository extension, so the calls are
identical apart from the port.

```bash
TRITON=http://tritonserver.neuriplo:8000
NEURIPLO=http://neuriplo-kserve-runtime.neuriplo:8080

# What is available? Everything staged, all UNAVAILABLE until asked for.
curl -X POST $TRITON/v2/repository/index
curl -X POST $NEURIPLO/v2/repository/index

# Load only what this request needs.
curl -X POST $TRITON/v2/repository/models/rfdetr-seg/load
curl -X POST $NEURIPLO/v2/repository/models/yolo-pose/load

# Infer.
curl -X POST $TRITON/v2/models/rfdetr-seg/infer   -d @seg-request.json
curl -X POST $NEURIPLO/v2/models/yolo-pose/infer  -d @pose-request.json

# Free the VRAM again. On a shared 6 GB card this is the part that matters.
curl -X POST $TRITON/v2/repository/models/rfdetr-seg/unload
curl -X POST $NEURIPLO/v2/repository/models/yolo-pose/unload
```

Unloading returns a model to the catalog rather than erasing it, so it still
appears in the index and can be loaded again.

A load that fails returns 409 with a reason. The common one is an engine built
against a different TensorRT than the server ships — the build succeeded in the
init container, so nothing failed at startup and the first load call is where it
surfaces.
