---
name: changelog
description: Update CHANGELOG.md with the work done on the current branch, following Keep a Changelog. Use when asked to update the changelog, prepare a release, or record what a branch changed.
---

# Changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Gather

```sh
git log --oneline develop..HEAD
git diff develop...HEAD --stat
```

Read the actual diff for anything the commit subjects leave ambiguous. Commit
messages here are short and imperative; they under-describe behavior changes.

## Write

Add entries under `## [Unreleased]`, in the right subsection — `Added`,
`Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`. Match the existing
voice: declarative, present tense, one behavior per bullet, wrapped at 79
columns.

An entry earns its place by telling an operator something that changes what they
would do. "Refactored internals" does not. "Refuse a non-empty existing ensemble
version" does.

Where a bullet explains a refusal or a default, say why in the same bullet — the
existing entries do this and it is the reason the file is readable a year later.

## Rules specific to this repository

- **Link every project on first mention**, in the changelog as everywhere else:
  [Triton](https://github.com/triton-inference-server/server),
  [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server),
  [KServe](https://github.com/kserve/kserve). A bare name is not a reference.
- Do not present the author's own projects — `neuriplo`,
  [neuriplo-kserve-runtime](https://github.com/olibartfast/neuriplo-kserve-runtime),
  this repository — as established third-party ones.
- Environment variables, defaults, and exit codes are the public contract. A
  change to any of them is a `Changed` entry even when nothing else moved.
- Record assertion-count changes in the suite, as existing entries do.
- Claim only what is measured. No performance claims without hardware and model
  size.

## Releasing

Only when explicitly asked. Move `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`,
open a fresh empty `Unreleased`, and update `VERSION`. A change to the
environment-variable contract, an exit code, or a default is not a patch bump.
