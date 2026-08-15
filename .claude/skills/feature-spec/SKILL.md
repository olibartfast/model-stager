---
name: feature-spec
description: Start the next phase of work on model-stager. Reads specs/roadmap.md for the first phase whose items are all unchecked, interviews the operator, creates the branch, and writes the dated feature spec directory. Use when asked to start the next phase, plan a feature, or begin roadmap work.
---

# Feature spec

Turn the next unstarted roadmap phase into a branch and a spec directory.

## 1. Find the phase

Read `specs/roadmap.md`. The next phase is the first `### Phase N — <name>`
section under `## Planned` in which **every** item is `[ ]`.

If the user named a phase, use that one instead. If every planned phase is
complete, say so and stop — do not invent a phase.

## 2. Branch

`AGENTS.md` mandates GitFlow. Branch from `develop`, never `master`:

```sh
git switch develop && git pull && git switch -c feature/phase-<N>-<kebab-name>
```

## 3. Interview — before writing anything to disk

Use `AskUserQuestion`, grouped on exactly these three:

- **Scope** — what this phase changes in `bin/model-stager`, which environment
  variables it adds, and what it deliberately leaves alone.
- **Decisions** — the open choices. Check `specs/tech-stack.md` "Open questions"
  for ones already recorded against this phase and put them here.
- **Context** — constraints, migration concerns for existing repositories, and
  what the operator would consider a regression.

Do not skip this because the roadmap phase already lists items. The roadmap says
what; the interview settles how.

## 4. Read the constitution

Before drafting, read `specs/mission.md`, `specs/tech-stack.md`, and `AGENTS.md`.

The invariants in `AGENTS.md` are not negotiable within a phase. Breaking one is
a design change that needs its own argument, not a task in a plan. In particular:
POSIX `sh` only, one file, server-neutral, backend-neutral by default,
model-agnostic, task-agnostic, local inputs only, fail loud, idempotent and
atomic, single writer.

## 5. Write `specs/YYYY-MM-DD-<phase-name>/`

Three files.

**`requirements.md`** — scope in and scope out; each decision with its rationale;
context and constraints; which layer of `specs/tech-stack.md` the change belongs
to and why it does not leak into another.

**`plan.md`** — numbered task groups, each with sub-tasks, ordered so the tree is
green between groups. Respect the ordering guarantee in `specs/tech-stack.md`:
admit → resolve → fingerprint → transform → publish → record.

**`validation.md`** — the assertions to add to `tests/test-model-stager.sh`; the
manual walkthrough; the done criteria. Per `AGENTS.md`, every new case must be
verified to fail against the unfixed code before it is committed — state how.

## 6. Constraints

- No new runtime dependency without explicit approval. The tool's tooling budget
  is POSIX utilities plus optional `trtexec`.
- Every project mentioned in any file gets a link on first mention, and this
  author's own projects (`neuriplo`, `neuriplo-kserve-runtime`, this repository)
  are never presented as established third-party ones. See `AGENTS.md`.
- Do not check off roadmap items in `specs/roadmap.md` until the phase merges.
