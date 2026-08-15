# Roadmap

Phased, independently shippable increments. The horizon is **operational
hardening**: making the guarantees in `specs/tech-stack.md` enforced rather than
documented, and making failure legible to the operator and the orchestrator.

Each phase is one branch off `develop`, one feature spec directory under
`specs/YYYY-MM-DD-<name>/`, and one entry in `CHANGELOG.md`.

## Current status

Shipped in `0.1.0` and on `Unreleased`. Both phases below are complete; they are
recorded so the roadmap describes the whole project, not only its future.

### Phase 0 — Extraction

- [x] Extract the TensorRT-to-`neuriplo` adapter from
      [neuriplo-kserve-runtime](https://github.com/olibartfast/neuriplo-kserve-runtime)
      into a standalone tool
- [x] Flat and tree staging shapes, mixable in one directory
- [x] Dispatch across ONNX, TensorRT, OpenVINO IR, TorchScript, TensorFlow,
      TFLite, ExecuTorch, DALI, and
      [Triton](https://github.com/triton-inference-server/server) ensembles
- [x] `REPOSITORY_LAYOUT` for `triton`, `ovms`
      ([OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)),
      and `neuriplo`
- [x] `Dockerfile` with swappable `BASE_IMAGE`; test suite with stubbed tooling

### Phase 1 — Generalization and verified idempotency

- [x] `MODEL_INPUTS` explicit input contract alongside catalog selection
- [x] Backend-neutral default: ONNX passes through; TensorRT is explicit
- [x] Fingerprint-verified warm starts; `config.pbtxt` outside version identity
- [x] Atomic publication of versions and configs
- [x] Model- and task-agnostic examples and manifests
- [x] Suite grown to 271 assertions, green under `dash` and BusyBox

---

## Planned

### Phase 2 — Mark the layers

Structural only. No behavior change, no new environment variable, no new test
*case* — the existing 271 assertions are the regression proof.

The layering in `specs/tech-stack.md` is real and unmarked. Hardening work lands
in the wrong place without it, and the "server-neutral" invariant cannot be
checked by reading.

- [ ] Add banner section comments to `bin/model-stager` for L0–L6, in dependency
      order
- [ ] State the ordering guarantee (admit → resolve → fingerprint → transform →
      publish → record) as a comment at the head of the orchestration layer
- [ ] Rename bare variables in `stage_flat` to the `stage_flat_*` prefix used
      elsewhere, closing the shadowing hazard
- [ ] Document `RESOLVED_TARGET` as the one deliberate out-parameter
- [ ] Confirm `shellcheck` clean and the suite green under `sh` and `dash`

### Phase 3 — Exit-code taxonomy

An init container that exits `1` for every cause forces `restartPolicy` to be
either uselessly aggressive or uselessly absent.

- [ ] Define and document codes: `2` invalid input/configuration (never retry),
      `3` missing or unreadable source (retry may succeed), `4` conflict with the
      published repository (operator action required), `5` required tooling absent
- [ ] Route every existing `exit 1` to a code
- [ ] Table in `README.md`; assertions per code in the suite
- [ ] Keep `1` reserved for unexpected failure

### Phase 4 — Enforce the single writer

Turn the strongest documented assumption into a checked one. The startup sweep of
`.stager-tmp.*` is destructive to a concurrent run; today nothing notices.

- [ ] Acquire an exclusive marker at the repository root before any sweep or write
- [ ] Record holder pid and start time; release on normal exit and on trap
- [ ] Break a stale marker only under an explicit, warning-emitting policy — a
      crashed stager must not deadlock the restart that replaces it
- [ ] Refuse with the Phase 3 conflict code when the marker is live
- [ ] Ensure the marker cannot be mistaken for a model by any supported layout

### Phase 5 — Verification cost control

The full-read guarantee is correct and, on a multi-gigabyte model over a remote
PVC, is paid on every restart. Give the operator the trade rather than the
default.

- [ ] `STAGER_VERIFY` with an explicit default matching today's behavior
- [ ] A cheaper level using size and mtime, escalating to checksum on mismatch
- [ ] A level that trusts recorded state without re-reading, for immutable volumes
- [ ] Never silently downgrade; the chosen level appears in the run summary
- [ ] Assertions that each level detects the drift it claims to, and that the
      cheap level's stated blind spots are documented

### Phase 6 — Dry run

- [ ] `STAGER_DRY_RUN=true` runs admission, dispatch resolution, and drift
      detection, writes nothing, and reports the tree it would produce
- [ ] Same refusals and same exit codes as a real run, so a dry run is a valid
      pre-flight check
- [ ] Assertions proving the repository is byte-identical after a dry run

### Phase 7 — Legible output

- [ ] A closing summary: models staged, skipped as unchanged, converted, refused
- [ ] Stable `error:` / `warning:` prefixes across every diagnostic
- [ ] A machine-readable manifest of the run, resolving open question 3 in
      `specs/tech-stack.md` first
- [ ] Assertions on the summary for a mixed run — some warm, some new, some
      converted

### Phase 8 — Adversarial test harness

Phases 4–6 make claims the current suite structurally cannot test.

- [ ] Harness support for a second concurrent stager against one repository
- [ ] Harness support for interrupting a run between transform and publish, and
      between the two metadata writes
- [ ] Cases: concurrent run refused; interrupted publish leaves no partial
      version; interrupted metadata write is retryable
- [ ] Per `AGENTS.md`, each new case verified to fail against the unfixed code

### Phase 9 — Operator documentation

- [ ] A runbook: what each exit code means and what to do about it
- [ ] Document the enforced single-writer contract in the Kubernetes examples —
      one staging Job or a `Recreate` rollout
- [ ] Guidance for choosing a `STAGER_VERIFY` level by volume type
- [ ] Reconcile `README.md`, `AGENTS.md`, and `specs/` after Phases 2–8

---

## Later phases (not yet planned)

Recorded so they are not mistaken for oversights:

- Additional formats or `REPOSITORY_LAYOUT` values — mechanical, per `AGENTS.md`,
  and driven by demand rather than scheduled
- Repository garbage collection (pruning superseded versions)
- Signature or provenance verification of source artifacts

## Explicitly out of scope

These are permanently rejected, not deferred. They appear here so a future
replanning does not relitigate them:

- Remote fetching of any kind (see `specs/mission.md`)
- Any interpretation of tensors, shapes, or task metadata
- A configuration file or DSL — tree-form input is the escape hatch
- Splitting `bin/model-stager` into sourced helpers
- Rewriting in a language with a runtime
