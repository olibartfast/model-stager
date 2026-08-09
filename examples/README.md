# Examples

| File | Scenario |
|---|---|
| [kubernetes/experiment-triton.yaml](kubernetes/experiment-triton.yaml) | Triton, staged repository, client-driven loading. |
| [kubernetes/experiment-neuriplo.yaml](kubernetes/experiment-neuriplo.yaml) | neuriplo-kserve-runtime, same, different layout. |
| [kubernetes/triton.yaml](kubernetes/triton.yaml) | Minimal single-server Triton, everything loaded at startup. |
| [kubernetes/ovms.yaml](kubernetes/ovms.yaml) | OpenVINO Model Server, no GPU; staging is copy-and-rename only. |
| [compose/docker-compose.yml](compose/docker-compose.yml) | The Triton flow without Kubernetes. |

## The two experiments

Two independent uses of the same stager image, proving the layout switch is the
only thing that changes between servers:

| | `experiment-triton` | `experiment-neuriplo` |
|---|---|---|
| namespace | `exp-triton` | `exp-neuriplo` |
| server | Triton 25.12 | neuriplo-kserve-runtime |
| `REPOSITORY_LAYOUT` | `triton` | `neuriplo` |
| loading | explicit | explicit |

### Model-agnostic by construction

Nothing in either manifest's *structure* — names, labels, probes, arguments,
volumes, ports — depends on which models are served. Exactly two lines per file
name a model:

```yaml
image: my-models:seg-v1                        # which models exist
- {name: MODELS, value: "rfdetr-seg,yolo26-seg"}   # which of them this one serves
```

Change either and nothing else changes. Leave `MODELS` unset and the deployment
stages everything the artifact image carries, naming no model at all.

That split is what lets one artifact image carry a catalogue while several
deployments each serve a slice of it. A name in `MODELS` that is not in the
staging directory is an error, not a smaller repository — serving three of four
requested models looks healthy from every angle except the client that needs the
fourth.

**Each is entirely self-contained** — its own namespace, PVC, Deployment and
Service. Applying one cannot reconfigure the other, and neither can touch a
deployment that already exists elsewhere in the cluster. Delete the namespace
and the experiment is gone.

Both are created with `replicas: 0`. Scale one up when you want to run it, and
back to 0 when finished. On a single-GPU node that is one at a time, unless you
have configured GPU sharing yourself.

```bash
kubectl apply -f kubernetes/experiment-triton.yaml
kubectl apply -f kubernetes/experiment-neuriplo.yaml

kubectl -n exp-triton scale deploy/triton --replicas=1
kubectl -n exp-triton logs -f deploy/triton -c model-stager   # engine build
# ... test ...
kubectl -n exp-triton scale deploy/triton --replicas=0

kubectl -n exp-neuriplo scale deploy/neuriplo --replicas=1
# ... test ...
kubectl -n exp-neuriplo scale deploy/neuriplo --replicas=0
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
SEG=http://triton.exp-triton:8000
POSE=http://neuriplo.exp-neuriplo:8080

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
