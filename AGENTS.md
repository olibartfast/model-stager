# Repository Guidelines

## What this repository is

`model-stager` turns a directory of exported model artifacts into a versioned
model repository tree (`<model>/<version>/<file>`) and exits. That tree is what
Triton, OpenVINO Model Server, and neuriplo-kserve-runtime all read.

`bin/model-stager` is the whole tool. `README.md` is the specification.

## Invariants

These are the reason the tool is reusable. Breaking one is a design change, not
a bug fix, and needs to be argued for explicitly:

- **Server-neutral.** No knowledge of any server's protocol, flags, or lifecycle.
  The only server-facing knowledge is the filename convention in
  `target_filename`, and adding a server means adding a case there.
- **Model-agnostic.** No model's name, tensor names, or shapes appear anywhere.
  A model's name comes from its staged filename; per-model settings arrive as
  environment suffixes whose values are the operator's business.
- **Task-agnostic.** Staging maps artifacts to filenames. It does not know what a
  model is *for*; task metadata is carried through `config.pbtxt` untouched, and
  never interpreted here.
- **POSIX `sh` only.** The container shell is often busybox `ash`. No bashisms —
  no arrays, no `local`, no `[[ ]]`, no `${var,,}`.
- **Single file.** `bin/model-stager` stays self-contained so it can be `COPY`d
  into any image without a directory of sourced helpers.
- **Fail loud.** An artifact that cannot be staged as asked is an error. Never
  substitute a different backend or silently skip a file — a dropped model
  surfaces much later as a 404, far from its cause.
- **Idempotent and atomic.** Staging re-runs on every restart. Skip finished
  work; publish a version by renaming a temp directory, never file by file.

## Build, Test, and Development Commands

```bash
tests/test-model-stager.sh                              # 97 assertions
shellcheck bin/model-stager tests/test-model-stager.sh
```

The suite needs no GPU, no TensorRT, and no server: `trtexec` and the server are
stubs on `PATH` that record how they were called. Every change to `bin/model-stager`
needs a case in `tests/test-model-stager.sh`.

Before pushing, run the suite under `dash` too, since CI does:

```bash
sudo ln -sf /bin/dash /bin/sh && tests/test-model-stager.sh
```

### Writing a test that is worth having

Check that a new test fails against the unfixed code before committing it. A
case can pass for a reason unrelated to what it claims to cover — the stale-temp
case originally asserted that a leftover directory was never published, which
was true whether or not the cleanup existed, because a temp directory carries
its creating run's pid and was never a publish candidate. It now asserts the
leftover is removed, which is what the cleanup actually does.

## Adding a format or a layout

1. Add a case to `target_filename` for each layout that can serve it, leaving
   the others empty so they are refused with a clear error.
2. Add an arm to the dispatch `case` in `stage_flat`, and to the recognized
   extension list in the main loop.
3. Add a test case, including the refusal for layouts that cannot serve it.
4. Update the dispatch and layout tables in `README.md`.

## MANDATORY: GitFlow Workflow

- `master` — production
- `develop` — integration branch for features
- `feature/*` or `fix/*` — branch from `develop`, merge to `develop` via PR

Do not commit feature work directly to `master`. Use PRs for merges into
`develop` and `master`.

## Commit & Pull Request Guidelines

Short imperative commit messages. Pull requests should summarize the change,
list validation commands run, and show the staged tree when dispatch behavior
changes.

## Security & Configuration Tips

Do not commit model files or engines — `.gitignore` covers the common
extensions, and a `.plan` is both large and useless on any other machine.
