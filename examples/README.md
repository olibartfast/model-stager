# Examples

| File | Scenario |
|---|---|
| [kubernetes/experiment-triton-seg.yaml](kubernetes/experiment-triton-seg.yaml) | Triton, TensorRT segmentation models, client-driven loading. |
| [kubernetes/experiment-neuriplo-pose.yaml](kubernetes/experiment-neuriplo-pose.yaml) | neuriplo-kserve-runtime, TensorRT pose models, client-driven loading. |
| [kubernetes/triton.yaml](kubernetes/triton.yaml) | Minimal single-server Triton, everything loaded at startup. |
| [kubernetes/ovms.yaml](kubernetes/ovms.yaml) | OpenVINO Model Server, no GPU; staging is copy-and-rename only. |
| [compose/docker-compose.yml](compose/docker-compose.yml) | The Triton flow without Kubernetes. |

## The two experiments

Two independent uses of the same stager image, proving the layout switch is the
only thing that changes between servers:

| | `experiment-triton-seg` | `experiment-neuriplo-pose` |
|---|---|---|
| namespace | `exp-triton-seg` | `exp-neuriplo-pose` |
| server | Triton 25.12 | neuriplo-kserve-runtime |
| models | `rfdetr-seg`, `yolo26-seg` | `rfdetr-pose`, `yolo-pose` |
| `REPOSITORY_LAYOUT` | `triton` | `neuriplo` |
| loading | explicit | explicit |

**Each is entirely self-contained** — its own namespace, PVC, Deployment and
Service. Applying one cannot reconfigure the other, and neither can touch a
deployment that already exists elsewhere in the cluster. Delete the namespace
and the experiment is gone.

Both are created with `replicas: 0`. Scale one up when you want to run it, and
back to 0 when finished. On a single-GPU node that is one at a time, unless you
have configured GPU sharing yourself.

```bash
kubectl apply -f kubernetes/experiment-triton-seg.yaml
kubectl apply -f kubernetes/experiment-neuriplo-pose.yaml

kubectl -n exp-triton-seg scale deploy/triton-seg --replicas=1
kubectl -n exp-triton-seg logs -f deploy/triton-seg -c model-stager   # engine build
# ... test ...
kubectl -n exp-triton-seg scale deploy/triton-seg --replicas=0

kubectl -n exp-neuriplo-pose scale deploy/neuriplo-pose --replicas=1
# ... test ...
kubectl -n exp-neuriplo-pose scale deploy/neuriplo-pose --replicas=0
```

The first scale-up of each builds its engines and takes minutes. The repository
is on a PVC, so later scale-ups skip the build and start in seconds.

### Readiness under explicit loading

`neuriplo-kserve-runtime` returns 503 from `/v2/health/ready` while no model is
loaded — measured:

```
$ neuriplo-kserve-runtime --models <repo> --model-control-mode explicit
/v2/health/live   -> HTTP 200
/v2/health/ready  -> HTTP 503
```

In explicit mode that is the normal startup state, so a `readinessProbe` on that
path never passes: the Service gets no endpoints, and nothing can reach the
server to issue the load that would clear it. The process is healthy and
unreachable.

That experiment therefore gates readiness on `/v2/health/live` — "the control
plane is up and accepting load requests", the only useful meaning of *ready* for
a server whose contents are the client's choice.

Triton with default `--strict-readiness` is ready once the server is up and every
*loaded* model is ready, vacuously true at zero models, so its probes stay on
`/v2/health/ready`. That is Triton's documented behaviour rather than something
measured here; if its rollout hangs at readiness, move those probes too.

### Driving either one from the client

Both servers speak the same model-repository extension, so the calls differ only
in address.

```bash
SEG=http://triton-seg.exp-triton-seg:8000
POSE=http://neuriplo-pose.exp-neuriplo-pose:8080

# Everything staged, all UNAVAILABLE until asked for.
curl -X POST $SEG/v2/repository/index
curl -X POST $POSE/v2/repository/index

# Load only what this request needs.
curl -X POST $SEG/v2/repository/models/rfdetr-seg/load
curl -X POST $POSE/v2/repository/models/yolo-pose/load

curl -X POST $SEG/v2/models/rfdetr-seg/infer   -d @seg-request.json
curl -X POST $POSE/v2/models/yolo-pose/infer   -d @pose-request.json

# Free the VRAM again.
curl -X POST $SEG/v2/repository/models/rfdetr-seg/unload
curl -X POST $POSE/v2/repository/models/yolo-pose/unload
```

Unloading returns a model to the catalog rather than erasing it, so it still
appears in the index and can be loaded again.

A failed load returns 409 with a reason. The common one is an engine built
against a different TensorRT than the server ships: the build succeeded in the
init container, so nothing failed at startup and the first load call is where it
surfaces.
