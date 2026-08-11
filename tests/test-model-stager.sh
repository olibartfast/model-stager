#!/usr/bin/env bash
set -uo pipefail

# Tests bin/model-stager.
#
# Runs with no GPU, no TensorRT, and no runtime binary: trtexec and
# model-server are replaced by stubs on PATH that record how they were
# called. What is under test is the dispatch and the tree it produces, not
# inference -- so this belongs in CI on an ordinary runner.

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PASS=0
FAIL=0

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASS=$((PASS + 1))
}
fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAIL=$((FAIL + 1))
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
STAGER="$REPO_DIR/bin/model-stager"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- stubs -------------------------------------------------------------------
# trtexec writes a marker file where a real engine would go and appends its
# argument line to a log, so a test can assert both that it ran and how.
BIN="$WORK/bin"
mkdir -p "$BIN"

cat >"$BIN/trtexec" <<'STUB'
#!/usr/bin/env bash
echo "$*" >>"$TRTEXEC_LOG"
if [ "${TRTEXEC_READ_STDIN:-false}" = "true" ]; then
    IFS= read -r consumed || true
fi
if [ "${TRTEXEC_FAIL:-false}" = "true" ]; then
    exit 1
fi
for arg in "$@"; do
    case "$arg" in
        --saveEngine=*) printf 'fake engine' >"${arg#--saveEngine=}" ;;
    esac
done
STUB

cat >"$BIN/model-server" <<'STUB'
#!/usr/bin/env bash
echo "$*" >>"$SERVE_LOG"
STUB

chmod +x "$BIN/trtexec" "$BIN/model-server"
export PATH="$BIN:$PATH"

# --- helpers -----------------------------------------------------------------
# Each case gets its own staging dir and repository so nothing leaks between
# them; run_stager returns the script's exit code without tripping set -e.
CASE=""
STAGE=""
REPO=""

new_case() {
    CASE="$1"
    STAGE="$WORK/$CASE/staging"
    REPO="$WORK/$CASE/repo"
    mkdir -p "$STAGE" "$REPO"
    export TRTEXEC_LOG="$WORK/$CASE/trtexec.log"
    export SERVE_LOG="$WORK/$CASE/serve.log"
    : >"$TRTEXEC_LOG"
    : >"$SERVE_LOG"
}

run_stager() {
    STAGE_DIR="$STAGE" MODEL_REPOSITORY="$REPO" REPOSITORY_LAYOUT="${REPOSITORY_LAYOUT:-neuriplo}" \
        ONNX_BACKEND="${ONNX_BACKEND:-tensorrt}" "$STAGER" "$@" \
        >"$WORK/$CASE/out.log" 2>&1
    return $?
}

expect_file() {
    if [ -f "$REPO/$1" ]; then
        pass "$CASE: $1 exists"
    else
        fail "$CASE: $1 missing"
        find "$REPO" -mindepth 1 | sed 's/^/      /'
    fi
}

expect_absent() {
    if [ -e "$REPO/$1" ]; then
        fail "$CASE: $1 should not exist"
    else
        pass "$CASE: $1 absent"
    fi
}

expect_exit() {
    if [ "$1" -eq "$2" ]; then
        pass "$CASE: exit $2"
    else
        fail "$CASE: expected exit $2, got $1"
        sed 's/^/      /' "$WORK/$CASE/out.log"
    fi
}

expect_log_contains() {
    if grep -qF -- "$2" "$1"; then
        pass "$CASE: log contains '$2'"
    else
        fail "$CASE: log missing '$2'"
        sed 's/^/      /' "$1"
    fi
}

expect_log_not_contains() {
    if grep -qF -- "$2" "$1"; then
        fail "$CASE: log unexpectedly contains '$2'"
        sed 's/^/      /' "$1"
    else
        pass "$CASE: log does not contain '$2'"
    fi
}

# =============================================================================
echo "=== Case 1: heterogeneous flat staging ==="
new_case heterogeneous
printf 'onnx' >"$STAGE/detector.onnx"
printf 'pte' >"$STAGE/ecdet.pte"
printf 'tflite' >"$STAGE/classifier.tflite"
printf 'xml' >"$STAGE/segmenter.xml"
printf 'bin' >"$STAGE/segmenter.bin"
printf 'dali' >"$STAGE/preprocess.dali"
printf 'graph' >"$STAGE/graph-model.json"
printf 'prebuilt' >"$STAGE/depth.plan"

SERVER_EXEC="model-server --models=$REPO" run_stager --port 8080
expect_exit $? 0
# The ONNX is compiled; every portable format is copied under a name whose
# extension selects its backend.
expect_file "detector/1/model.plan"
expect_file "ecdet/1/model.pte"
expect_file "classifier/1/model.tflite"
expect_file "segmenter/1/model.xml"
expect_file "segmenter/1/model.bin"
expect_file "preprocess/1/model.dali"
expect_file "graph-model/1/model.json"
expect_file "depth/1/model.plan"
expect_log_contains "$SERVE_LOG" "--models=$REPO"
expect_log_contains "$SERVE_LOG" "--port 8080"

# =============================================================================
echo "=== Case 2: OpenVINO weights are renamed to match the .xml ==="
new_case openvino
printf 'xml' >"$STAGE/seg.xml"
printf 'weights' >"$STAGE/seg.bin"
run_stager
expect_exit $? 0
# OpenVINO resolves weights by basename, so seg.bin must land as model.bin.
if [ "$(cat "$REPO/seg/1/model.bin" 2>/dev/null)" = "weights" ]; then
    pass "$CASE: .bin renamed alongside .xml"
else
    fail "$CASE: .bin not renamed to model.bin"
fi
expect_absent "seg/1/seg.bin"

# =============================================================================
echo "=== Case 3: orphan .bin is an error, not a silent skip ==="
new_case orphan-bin
printf 'weights' >"$STAGE/lonely.bin"
run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "no matching .xml"

# =============================================================================
echo "=== Case 4: unrecognized artifact fails loudly, opt-out available ==="
new_case unknown
printf 'onnx' >"$STAGE/good.onnx"
printf 'labels' >"$STAGE/labels.names"
run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "unrecognized staged file"

new_case unknown-ignored
printf 'onnx' >"$STAGE/good.onnx"
printf 'labels' >"$STAGE/labels.names"
STAGER_IGNORE_UNKNOWN=true run_stager
expect_exit $? 0
expect_file "good/1/model.plan"

# =============================================================================
echo "=== Case 5: ONNX_BACKEND=onnx_runtime skips compilation ==="
new_case onnx-passthrough
printf 'onnx' >"$STAGE/detector.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "detector/1/model.onnx"
expect_absent "detector/1/model.plan"
if [ -s "$TRTEXEC_LOG" ]; then
    fail "$CASE: trtexec ran despite ONNX_BACKEND=onnx_runtime"
else
    pass "$CASE: trtexec never invoked"
fi

# =============================================================================
echo "=== Case 6: preparation is idempotent across restarts ==="
new_case idempotent
printf 'onnx' >"$STAGE/detector.onnx"
run_stager
expect_exit $? 0
first_count=$(wc -l <"$TRTEXEC_LOG")
run_stager
expect_exit $? 0
second_count=$(wc -l <"$TRTEXEC_LOG")
if [ "$first_count" -eq 1 ] && [ "$second_count" -eq 1 ]; then
    pass "$CASE: engine built once, reused on restart"
else
    fail "$CASE: trtexec ran $second_count times across two starts (expected 1)"
fi

# =============================================================================
echo "=== Case 7: a failed conversion publishes nothing ==="
new_case atomic
printf 'onnx' >"$STAGE/detector.onnx"
TRTEXEC_FAIL=true run_stager
expect_exit $? 1
# The half-built directory must not look like a finished model to the next start.
expect_absent "detector/1"

# A retry after the failure succeeds, rather than tripping over leftovers.
run_stager
expect_exit $? 0
expect_file "detector/1/model.plan"

# =============================================================================
echo "=== Case 8: a stale temp directory never gets published ==="
new_case stale-tmp
printf 'onnx' >"$STAGE/detector.onnx"
mkdir -p "$REPO/detector/.stager-tmp.1.999"
printf 'truncated' >"$REPO/detector/.stager-tmp.1.999/model.plan"
run_stager
expect_exit $? 0
expect_file "detector/1/model.plan"
if [ "$(cat "$REPO/detector/1/model.plan")" = "fake engine" ]; then
    pass "$CASE: published the rebuilt engine, not the leftover"
else
    fail "$CASE: leftover temp content was published"
fi
# The leftover is swept, not merely ignored. A temp directory from a killed run
# is never a publish candidate anyway -- it carries that run's pid -- so without
# this assertion the case would pass with the cleanup removed entirely, and
# orphans would accumulate on the volume across every restart.
expect_absent "detector/.stager-tmp.1.999"

# =============================================================================
echo "=== Case 9: tree-form staging is copied verbatim ==="
new_case tree-form
mkdir -p "$STAGE/tree-model/3"
printf 'engine' >"$STAGE/tree-model/3/model.plan"
printf 'labels' >"$STAGE/tree-model/3/labels.txt"
printf 'config' >"$STAGE/tree-model/config.pbtxt"
run_stager
expect_exit $? 0
# Verbatim means the version and the extra files survive: this is the escape
# hatch for anything the flat form cannot express.
expect_file "tree-model/3/model.plan"
expect_file "tree-model/3/labels.txt"
expect_file "tree-model/config.pbtxt"
expect_absent "tree-model/1"

# =============================================================================
echo "=== Case 10: flat .pbtxt lands beside the version directory ==="
new_case pbtxt-overlay
printf 'pte' >"$STAGE/ecdet.pte"
printf 'input { name: "images" }' >"$STAGE/ecdet.pbtxt"
run_stager
expect_exit $? 0
expect_file "ecdet/1/model.pte"
expect_file "ecdet/config.pbtxt"
expect_absent "ecdet/1/model.pbtxt"

new_case pbtxt-orphan
printf 'config' >"$STAGE/nobody.pbtxt"
run_stager
expect_exit $? 1

# =============================================================================
echo "=== Case 11: per-model TRT overrides ==="
new_case per-model-shapes
printf 'onnx' >"$STAGE/static-model.onnx"
printf 'onnx' >"$STAGE/dynamic-model.onnx"
# One static model and one dynamic model in the same repository: a single global
# TRT_SHAPES cannot express this, which is what the override exists for.
TRT_SHAPES_DYNAMIC_MODEL="input:1x3x480x640" TRT_PRECISION_STATIC_MODEL=fp32 run_stager
expect_exit $? 0
expect_log_contains "$TRTEXEC_LOG" "--shapes=input:1x3x480x640"
if grep -F -- "static-model" "$TRTEXEC_LOG" | grep -qF -- "--shapes="; then
    fail "$CASE: static model was given explicit shapes"
else
    pass "$CASE: static model built without --shapes"
fi
if grep -F -- "static-model" "$TRTEXEC_LOG" | grep -qF -- "--fp16"; then
    fail "$CASE: TRT_PRECISION_STATIC_MODEL=fp32 did not override the default"
else
    pass "$CASE: per-model precision override applied"
fi

# =============================================================================
echo "=== Case 12: empty staging refuses to serve ==="
new_case empty
run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "no servable artifacts staged"
if [ -s "$SERVE_LOG" ]; then
    fail "$CASE: server started on an empty repository"
else
    pass "$CASE: server never started"
fi

# =============================================================================
echo "=== Case 13: missing trtexec is not a silent substitution ==="
new_case no-trtexec
printf 'onnx' >"$STAGE/detector.onnx"
mv "$BIN/trtexec" "$BIN/trtexec.hidden"
run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "trtexec not found"

new_case no-trtexec-fallback
printf 'onnx' >"$STAGE/detector.onnx"
TRT_FALLBACK_ONNX=true run_stager
expect_exit $? 0
expect_file "detector/1/model.onnx"
mv "$BIN/trtexec.hidden" "$BIN/trtexec"

# =============================================================================
echo "=== Case 14: staging and exiting is the default ==="
new_case stage-and-exit
printf 'onnx' >"$STAGE/detector.onnx"
printf 'pte' >"$STAGE/ecdet.pte"
run_stager
expect_exit $? 0
expect_file "detector/1/model.plan"
expect_file "ecdet/1/model.pte"
# The whole point of an init container: the tree is the deliverable, and nothing
# is launched by the thing that built it. This must hold with no opt-in, because
# a tool that starts a server unless told otherwise is not a staging step.
if [ -s "$SERVE_LOG" ]; then
    fail "$CASE: a server was started without SERVER_EXEC"
else
    pass "$CASE: no server started"
fi

# =============================================================================
echo "=== Case 15: SERVER_EXEC drives an arbitrary server ==="
new_case server-exec
printf 'onnx' >"$STAGE/detector.onnx"
cat >"$BIN/tritonserver" <<'STUB'
#!/usr/bin/env bash
echo "$*" >>"$SERVE_LOG"
STUB
chmod +x "$BIN/tritonserver"
SERVER_EXEC="tritonserver --model-repository=$REPO" run_stager --strict-readiness=true
expect_exit $? 0
expect_log_contains "$SERVE_LOG" "--model-repository=$REPO"
# Container arguments still reach the server, so a manifest's args: keep working
# whichever server is wrapped.
expect_log_contains "$SERVE_LOG" "--strict-readiness=true"

# =============================================================================
echo "=== Case 16: Triton layout renames to Triton's conventions ==="
new_case triton-layout
printf 'onnx' >"$STAGE/detector.onnx"
printf 'ts' >"$STAGE/script-model.torchscript"
printf 'graph' >"$STAGE/legacy.pb"
printf 'xml' >"$STAGE/seg.xml"
printf 'bin' >"$STAGE/seg.bin"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_file "detector/1/model.plan"
# Triton detects platform by filename, so a TorchScript must be model.pt and a
# frozen graph model.graphdef whatever the staged extension was.
expect_file "script-model/1/model.pt"
expect_absent "script-model/1/model.torchscript"
expect_file "legacy/1/model.graphdef"
expect_absent "legacy/1/model.pb"
expect_file "seg/1/model.xml"
expect_file "seg/1/model.bin"

# =============================================================================
echo "=== Case 17: OVMS layout, and formats it cannot serve are refused ==="
new_case ovms-layout
printf 'xml' >"$STAGE/seg.xml"
printf 'bin' >"$STAGE/seg.bin"
REPOSITORY_LAYOUT=ovms ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "seg/1/model.xml"
expect_file "seg/1/model.bin"

new_case ovms-refuses-engine
printf 'onnx' >"$STAGE/detector.onnx"
# OVMS cannot load a TensorRT engine. Compiling one and dropping it in the tree
# would produce a model that silently never appears, so this must fail here.
REPOSITORY_LAYOUT=ovms run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "cannot serve a .plan model"
# Refusing must leave the repository untouched, not a half-built model directory
# that the next start would mistake for finished work.
expect_absent "detector/1"
if [ -s "$TRTEXEC_LOG" ]; then
    fail "$CASE: compiled an engine for a layout that cannot load one"
else
    pass "$CASE: refused before doing any work"
fi

new_case triton-refuses-executorch
printf 'pte' >"$STAGE/ecdet.pte"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "cannot serve a .pte model"

# =============================================================================
echo "=== Case 18: an unknown layout is rejected up front ==="
new_case bad-layout
printf 'onnx' >"$STAGE/detector.onnx"
REPOSITORY_LAYOUT=torchserve run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "unsupported REPOSITORY_LAYOUT"

# =============================================================================
echo "=== Case 19: dynamic models get an optimization profile ==="
new_case dynamic-profile
printf 'onnx' >"$STAGE/dynamic-model.onnx"
TRT_MIN_SHAPES="input1:1x3x256x256" \
    TRT_OPT_SHAPES="input1:1x3x520x960" \
    TRT_MAX_SHAPES="input1:1x3x1080x1920" run_stager
expect_exit $? 0
expect_file "dynamic-model/1/model.plan"
# A dynamic model needs min/opt/max together. --shapes builds no profile, so an
# engine built with it cannot accept the range it was supposed to serve.
expect_log_contains "$TRTEXEC_LOG" "--minShapes=input1:1x3x256x256"
expect_log_contains "$TRTEXEC_LOG" "--optShapes=input1:1x3x520x960"
expect_log_contains "$TRTEXEC_LOG" "--maxShapes=input1:1x3x1080x1920"
if grep -qF -- "--shapes=" "$TRTEXEC_LOG"; then
    fail "$CASE: passed --shapes alongside a profile"
else
    pass "$CASE: no --shapes when a profile is given"
fi

new_case partial-profile
printf 'onnx' >"$STAGE/dynamic-model.onnx"
# Two of three is not a profile. Accepting it would build an engine whose
# bounds are not the ones anyone asked for.
TRT_MIN_SHAPES="input1:1x3x256x256" TRT_MAX_SHAPES="input1:1x3x1080x1920" run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "together"
if [ -s "$TRTEXEC_LOG" ]; then
    fail "$CASE: started a build with an incomplete profile"
else
    pass "$CASE: refused before building"
fi

new_case conflicting-shapes
printf 'onnx' >"$STAGE/dynamic-model.onnx"
TRT_SHAPES="input1:1x3x520x960" \
    TRT_MIN_SHAPES="input1:1x3x256x256" \
    TRT_OPT_SHAPES="input1:1x3x520x960" \
    TRT_MAX_SHAPES="input1:1x3x1080x1920" run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "both TRT_SHAPES and a min/opt/max profile"

new_case per-model-profile
printf 'onnx' >"$STAGE/dynamic-model.onnx"
printf 'onnx' >"$STAGE/static-model.onnx"
# The real mixed case: one dynamic model needs a profile, and the static one
# beside it must be built with no shape arguments at all.
TRT_MIN_SHAPES_DYNAMIC_MODEL="input1:1x3x256x256" \
    TRT_OPT_SHAPES_DYNAMIC_MODEL="input1:1x3x520x960" \
    TRT_MAX_SHAPES_DYNAMIC_MODEL="input1:1x3x1080x1920" run_stager
expect_exit $? 0
expect_file "dynamic-model/1/model.plan"
expect_file "static-model/1/model.plan"
if grep -F -- "static-model" "$TRTEXEC_LOG" | grep -qE -- "--(min|opt|max)Shapes="; then
    fail "$CASE: static model was given a profile"
else
    pass "$CASE: profile applied only to the dynamic model"
fi

# =============================================================================
echo "=== Case 20: a GPU pre/post-processing ensemble stages end to end ==="
new_case gpu-ensemble
# The real shape of a GPU pipeline: DALI decodes and resizes on device, TensorRT
# infers, and the ensemble wires them together. All three land in one repository.
printf 'dali' >"$STAGE/preprocess.dali"
printf 'onnx' >"$STAGE/detector.onnx"
printf 'dali' >"$STAGE/postprocess.dali"
cat >"$STAGE/pipeline.pbtxt" <<'PBTXT'
name: "pipeline"
platform: "ensemble"
ensemble_scheduling {
  step [
    { model_name: "preprocess"  model_version: -1 },
    { model_name: "detector"    model_version: -1 },
    { model_name: "postprocess" model_version: -1 }
  ]
}
PBTXT
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_file "preprocess/1/model.dali"
expect_file "detector/1/model.plan"
expect_file "postprocess/1/model.dali"
# An ensemble has no model file: the graph is the config, and Triton requires the
# version directory to exist but be empty.
expect_file "pipeline/config.pbtxt"
if [ -d "$REPO/pipeline/1" ] && [ -z "$(ls -A "$REPO/pipeline/1")" ]; then
    pass "$CASE: ensemble version directory exists and is empty"
else
    fail "$CASE: ensemble version directory wrong"
    find "$REPO/pipeline" | sed 's/^/      /'
fi

# =============================================================================
echo "=== Case 21: an ensemble survives restarts unchanged ==="
new_case ensemble-restart
printf 'platform: "ensemble"\nensemble_scheduling { step [] }\n' >"$STAGE/pipeline.pbtxt"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
# Regression: an empty version directory read as "not staged", so every restart
# re-staged it -- and `mv tmp dir` moves tmp *into* an existing dir, nesting one
# temp directory per restart inside the ensemble's version.
nested=$(find "$REPO/pipeline" -name '.stager-tmp.*' | wc -l)
if [ "$nested" -eq 0 ]; then
    pass "$CASE: no temp directory nested by restarts"
else
    fail "$CASE: $nested temp directories nested inside the ensemble"
    find "$REPO/pipeline" | sed 's/^/      /'
fi

# =============================================================================
echo "=== Case 22: a tree-form ensemble is staged as written ==="
new_case ensemble-tree
mkdir -p "$STAGE/seg_pipeline/2"
printf 'platform: "ensemble"\n' >"$STAGE/seg_pipeline/config.pbtxt"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_file "seg_pipeline/config.pbtxt"
if [ -d "$REPO/seg_pipeline/2" ]; then
    pass "$CASE: tree-form ensemble keeps its version"
else
    fail "$CASE: version directory missing"
fi

# =============================================================================
echo "=== Case 23: an orphaned overlay is still an error ==="
new_case overlay-typo
printf 'onnx' >"$STAGE/detector.onnx"
# Not an ensemble, and names no staged model: almost always a basename typo, and
# accepting it would leave a config nothing reads.
printf 'input { name: "images" }\n' >"$STAGE/detektor.pbtxt"
run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "no matching model in the repository"

# =============================================================================
echo "=== Case 24: neuriplo ensembles stay graph files ==="
new_case neuriplo-ensemble
printf 'dali' >"$STAGE/preprocess.dali"
printf '{"steps":[]}' >"$STAGE/graph-model.json"
run_stager
expect_exit $? 0
expect_file "preprocess/1/model.dali"
expect_file "graph-model/1/model.json"

# =============================================================================
echo "=== Case 25: MODELS selects a subset of the artifact image ==="
new_case models-subset
# One artifact image carrying a catalogue; this deployment serves two of it.
printf 'onnx' >"$STAGE/catalog-a.onnx"
printf 'onnx' >"$STAGE/catalog-b.onnx"
printf 'onnx' >"$STAGE/catalog-c.onnx"
printf 'onnx' >"$STAGE/catalog-d.onnx"
MODELS="catalog-c,catalog-d" run_stager
expect_exit $? 0
expect_file "catalog-c/1/model.plan"
expect_file "catalog-d/1/model.plan"
expect_absent "catalog-a"
expect_absent "catalog-b"
# Unselected models must cost nothing: an engine build is minutes.
if [ "$(wc -l <"$TRTEXEC_LOG")" -eq 2 ]; then
    pass "$CASE: built only the selected models"
else
    fail "$CASE: built $(wc -l <"$TRTEXEC_LOG") engines, expected 2"
fi

new_case models-spaces
printf 'onnx' >"$STAGE/a.onnx"
printf 'onnx' >"$STAGE/b.onnx"
MODELS="a b" run_stager
expect_exit $? 0
expect_file "a/1/model.plan"
expect_file "b/1/model.plan"

# =============================================================================
echo "=== Case 26: a requested model that is not staged is an error ==="
new_case models-missing
printf 'onnx' >"$STAGE/catalog-a.onnx"
# Serving three of four looks healthy everywhere except the client that needs
# the fourth, so a typo or a stale artifact image has to fail here.
MODELS="catalog-a,catalog-missing" run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "requested but not staged: catalog-missing"

# =============================================================================
echo "=== Case 27: MODELS unset still stages everything ==="
new_case models-unset
printf 'onnx' >"$STAGE/a.onnx"
printf 'pte' >"$STAGE/b.pte"
run_stager
expect_exit $? 0
expect_file "a/1/model.plan"
expect_file "b/1/model.pte"

# =============================================================================
echo "=== Case 28: MODELS selects tree-form models too ==="
new_case models-tree
mkdir -p "$STAGE/wanted/1" "$STAGE/unwanted/1"
printf 'engine' >"$STAGE/wanted/1/model.plan"
printf 'engine' >"$STAGE/unwanted/1/model.plan"
MODELS=wanted run_stager
expect_exit $? 0
expect_file "wanted/1/model.plan"
expect_absent "unwanted"

# =============================================================================
echo "=== Case 29: MODELS ignores configs belonging to unselected models ==="
new_case models-config-subset
printf 'onnx' >"$STAGE/wanted.onnx"
printf 'input { name: "x" }\n' >"$STAGE/wanted.pbtxt"
printf 'onnx' >"$STAGE/unwanted.onnx"
printf 'input { name: "x" }\n' >"$STAGE/unwanted.pbtxt"
MODELS=wanted ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "wanted/1/model.onnx"
expect_file "wanted/config.pbtxt"
expect_absent "unwanted"

# =============================================================================
echo "=== Case 30: MODELS can select a config-only Triton ensemble ==="
new_case models-ensemble
printf 'platform: "ensemble"\nensemble_scheduling {}\n' >"$STAGE/pipeline.pbtxt"
MODELS=pipeline REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_file "pipeline/config.pbtxt"
if [ -d "$REPO/pipeline/1" ]; then
    pass "$CASE: empty ensemble version exists"
else
    fail "$CASE: empty ensemble version missing"
fi

# =============================================================================
echo "=== Case 31: non-Triton layouts refuse a flat Triton ensemble ==="
new_case ovms-refuses-triton-ensemble
printf 'platform: "ensemble"\nensemble_scheduling {}\n' >"$STAGE/pipeline.pbtxt"
REPOSITORY_LAYOUT=ovms run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "cannot serve a Triton ensemble"
expect_absent "pipeline"

# =============================================================================
echo "=== Case 32: conflicting flat artifacts fail instead of picking one ==="
new_case duplicate-model
printf 'onnx' >"$STAGE/model.onnx"
printf 'engine' >"$STAGE/model.plan"
REPOSITORY_LAYOUT=triton ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "multiple artifacts for model 'model'"
expect_absent "model"

# =============================================================================
echo "=== Case 33: ONNX pass-through is the backend-neutral default ==="
new_case neutral-onnx-default
printf 'onnx' >"$STAGE/model.onnx"
STAGE_DIR="$STAGE" MODEL_REPOSITORY="$REPO" REPOSITORY_LAYOUT=triton "$STAGER" \
    >"$WORK/$CASE/out.log" 2>&1
expect_exit $? 0
expect_file "model/1/model.onnx"
expect_absent "model/1/model.plan"
if [ -s "$TRTEXEC_LOG" ]; then
    fail "$CASE: default invoked trtexec"
else
    pass "$CASE: default used no conversion backend"
fi

# =============================================================================
echo "=== Case 34: a mounted single-model repository is recognized ==="
new_case single-model-root
mkdir -p "$STAGE/7"
printf 'onnx' >"$STAGE/7/model.onnx"
printf 'backend: "onnxruntime"\n' >"$STAGE/config.pbtxt"
MODELS=mounted-model REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_file "mounted-model/7/model.onnx"
expect_file "mounted-model/config.pbtxt"

# =============================================================================
echo "=== Case 35: MODEL_INPUTS maps names to independent local inputs ==="
new_case explicit-inputs
mkdir -p "$WORK/$CASE/inputs/tree/3"
printf 'onnx' >"$WORK/$CASE/inputs/flat.onnx"
printf 'plan' >"$WORK/$CASE/inputs/tree/3/model.plan"
MODEL_INPUTS="flat=$WORK/$CASE/inputs/flat.onnx
tree=$WORK/$CASE/inputs/tree" \
    REPOSITORY_LAYOUT=triton ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "flat/1/model.onnx"
expect_file "tree/3/model.plan"

new_case explicit-inputs-missing
MODEL_INPUTS="missing=$WORK/$CASE/does-not-exist" run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "input path not found"

# =============================================================================
echo "=== Case 36: unsafe flat-form versions are rejected ==="
new_case unsafe-version
printf 'onnx' >"$STAGE/model.onnx"
MODEL_VERSION=../../outside ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "invalid MODEL_VERSION"
expect_absent "model"

# =============================================================================
echo "=== Case 37: ambiguous directories fail instead of being guessed ==="
new_case ambiguous-directory
mkdir -p "$STAGE/model/variables"
printf 'weights' >"$STAGE/model/variables/data"
run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "not one repository-shaped model"
expect_absent "model"

# =============================================================================
echo "=== Case 38: an existing repository can be validated in place ==="
new_case repository-in-place
mkdir -p "$REPO/model-a/1" "$REPO/model-b/2"
printf 'a' >"$REPO/model-a/1/model.onnx"
printf 'b' >"$REPO/model-b/2/model.onnx"
STAGE="$REPO"
run_stager
expect_exit $? 0
expect_file "model-a/1/model.onnx"
expect_file "model-b/2/model.onnx"

MODELS=model-a run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "cannot filter MODELS in place"

# =============================================================================
echo "=== Case 39: explicit model names must be unique ==="
new_case duplicate-explicit-input
printf 'onnx' >"$STAGE/a.onnx"
printf 'onnx' >"$STAGE/b.onnx"
MODEL_INPUTS="same=$STAGE/a.onnx
same=$STAGE/b.onnx" ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "multiple MODEL_INPUTS records"
expect_absent "same"

# =============================================================================
echo "=== Case 40: stale atomic-config files are swept ==="
new_case stale-config
mkdir -p "$REPO/model"
printf 'partial' >"$REPO/model/.stager-config.999"
printf 'onnx' >"$STAGE/model.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "model/1/model.onnx"
expect_absent "model/.stager-config.999"

# =============================================================================
echo "=== Case 41: warm flat versions must match the requested staging ==="
new_case warm-version-contract
printf 'torchscript' >"$STAGE/model-b.torchscript"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_file "model-b/1/model.pt"
REPOSITORY_LAYOUT=neuriplo run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "existing version does not match requested staging"

new_case warm-source-drift
printf 'first' >"$STAGE/model.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
printf 'second' >"$STAGE/model.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "existing version does not match requested staging"

new_case warm-backend-switch
printf 'onnx' >"$STAGE/model.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
ONNX_BACKEND=tensorrt run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "existing version does not match requested staging"

new_case warm-output-drift
printf 'onnx' >"$STAGE/model.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
printf 'tampered' >"$REPO/model/1/model.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "existing version does not match requested staging"

new_case warm-config-drift
printf 'onnx' >"$STAGE/model.onnx"
printf 'name: "model"\n' >"$STAGE/model.pbtxt"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
printf 'name: "model"\nmax_batch_size: 8\n' >"$STAGE/model.pbtxt"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_log_contains "$REPO/model/config.pbtxt" "max_batch_size: 8"

new_case warm-overlay-only-drift
printf 'onnx' >"$STAGE/model.onnx"
printf 'name: "model"\n' >"$STAGE/model.pbtxt"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
rm "$STAGE/model.onnx"
printf 'name: "model"\nmax_batch_size: 8\n' >"$STAGE/model.pbtxt"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_log_contains "$REPO/model/config.pbtxt" "max_batch_size: 8"

new_case warm-tree-drift
mkdir -p "$STAGE/model/3"
printf 'plan' >"$STAGE/model/3/model.plan"
run_stager
expect_exit $? 0
printf 'changed' >"$STAGE/model/3/model.plan"
run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "does not match repository-shaped input"

new_case warm-tree-config-update
mkdir -p "$STAGE/model/3"
printf 'plan' >"$STAGE/model/3/model.plan"
printf 'name: "model"\n' >"$STAGE/model/config.pbtxt"
run_stager
expect_exit $? 0
printf 'name: "model"\nmax_batch_size: 8\n' >"$STAGE/model/config.pbtxt"
run_stager
expect_exit $? 0
expect_log_contains "$REPO/model/config.pbtxt" "max_batch_size: 8"

new_case warm-ensemble-config-drift
printf 'platform: "ensemble"\nensemble_scheduling {}\n' >"$STAGE/pipeline.pbtxt"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
printf 'platform: "ensemble"\nensemble_scheduling { step [] }\n' >"$STAGE/pipeline.pbtxt"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_log_contains "$REPO/pipeline/config.pbtxt" "step []"

new_case adopt-untracked-onnx
printf 'onnx' >"$STAGE/model.onnx"
mkdir -p "$REPO/model/1"
printf 'onnx' >"$REPO/model/1/model.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "model/1/.model-stager-state"
expect_log_contains "$WORK/$CASE/out.log" "adopting untracked existing version"

new_case adopt-untracked-engine
printf 'onnx' >"$STAGE/model.onnx"
mkdir -p "$REPO/model/1"
printf 'old engine' >"$REPO/model/1/model.plan"
ONNX_BACKEND=tensorrt run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "cannot verify existing TensorRT engine"
STAGER_ADOPT_UNVERIFIED_TENSORRT=true ONNX_BACKEND=tensorrt run_stager
expect_exit $? 0
expect_file "model/1/.model-stager-state"
expect_log_contains "$WORK/$CASE/out.log" "WARNING: adopting unverified TensorRT engine"
if [ -s "$TRTEXEC_LOG" ]; then
    fail "$CASE: adoption rebuilt the existing engine"
else
    pass "$CASE: adoption did not rebuild the existing engine"
fi

new_case flat-to-ensemble
printf 'onnx' >"$STAGE/model.onnx"
REPOSITORY_LAYOUT=triton ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
rm "$STAGE/model.onnx"
printf 'platform: "ensemble"\nensemble_scheduling {}\n' >"$STAGE/model.pbtxt"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "ensemble version directory is not empty"

new_case equivalent-layout-identity
printf 'plan' >"$STAGE/model.plan"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
REPOSITORY_LAYOUT=neuriplo run_stager
expect_exit $? 0
expect_log_contains "$WORK/$CASE/out.log" "already staged and verified"

new_case equivalent-extension-identity
printf 'script' >"$STAGE/model.torchscript"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
rm "$STAGE/model.torchscript"
printf 'script' >"$STAGE/model.pt"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_log_contains "$WORK/$CASE/out.log" "already staged and verified"

new_case stale-metadata-temp
mkdir -p "$REPO/model/1"
printf 'onnx' >"$STAGE/model.onnx"
printf 'onnx' >"$REPO/model/1/model.onnx"
printf 'partial' >"$REPO/model/1/.model-stager-state.9999"
printf 'partial' >"$REPO/model/1/.model-stager-output.9999"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_absent "model/1/.model-stager-state.9999"
expect_absent "model/1/.model-stager-output.9999"

# =============================================================================
echo "=== Case 42: opaque configs prevent unsafe model renames ==="
new_case explicit-config-rename
printf 'onnx' >"$STAGE/detector.onnx"
printf 'name: "detector"\n' >"$STAGE/detector.pbtxt"
MODEL_INPUTS="vision=$STAGE/detector.onnx" ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "configured input cannot be renamed"
expect_absent "vision"

new_case tree-config-rename
mkdir -p "$STAGE/1"
printf 'onnx' >"$STAGE/1/model.onnx"
printf 'name: "detector"\n' >"$STAGE/config.pbtxt"
MODELS=vision run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "configured input cannot be renamed"
expect_absent "vision"

# =============================================================================
echo "=== Case 43: explicit sibling configs are part of the input contract ==="
new_case explicit-sibling-config
printf 'onnx' >"$STAGE/model.onnx"
printf 'name: "model"\n' >"$STAGE/model.pbtxt"
MODEL_INPUTS="model=$STAGE/model.onnx" ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "model/1/model.onnx"
expect_file "model/config.pbtxt"

# =============================================================================
echo "=== Case 44: explicit mode diagnostics name the input mode ==="
new_case blank-explicit-inputs
MODEL_INPUTS='

' run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "MODEL_INPUTS contains no model records"

# =============================================================================
echo "=== Case 45: conversion tools cannot consume the explicit input list ==="
new_case explicit-stdin-isolation
printf 'onnx' >"$STAGE/a.onnx"
printf 'plan' >"$STAGE/b.plan"
TRTEXEC_READ_STDIN=true MODEL_INPUTS="a=$STAGE/a.onnx
b=$STAGE/b.plan" ONNX_BACKEND=tensorrt run_stager
expect_exit $? 0
expect_file "a/1/model.plan"
expect_file "b/1/model.plan"

# =============================================================================
echo "=== Case 46: OVMS accepts TensorFlow Lite artifacts ==="
new_case ovms-tflite
printf 'tflite' >"$STAGE/model.tflite"
REPOSITORY_LAYOUT=ovms run_stager
expect_exit $? 0
expect_file "model/1/model.tflite"

# =============================================================================
echo "=== Case 47: catalog model names and hidden entries are validated ==="
new_case invalid-catalog-name
printf 'onnx' >"$STAGE/model a.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "invalid model name"
expect_absent "model a"

new_case hidden-catalog-entry
printf 'onnx' >"$STAGE/.hidden.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "hidden catalog entry"

new_case unselected-invalid-name
printf 'onnx' >"$STAGE/wanted.onnx"
printf 'onnx' >"$STAGE/odd name.onnx"
MODELS=wanted ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "wanted/1/model.onnx"
expect_absent "odd name"

# =============================================================================
echo "=== Case 48: nested tree inputs and override namespaces fail loud ==="
new_case hidden-tree-entry
mkdir -p "$STAGE/model/1"
printf 'onnx' >"$STAGE/model/1/model.onnx"
printf 'extra' >"$STAGE/model/.extra_weights.bin"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "hidden model entry is not stageable"
expect_absent "model"

new_case hidden-tree-entry-ignored
mkdir -p "$STAGE/model/1"
printf 'onnx' >"$STAGE/model/1/model.onnx"
printf 'metadata' >"$STAGE/model/.DS_Store"
STAGER_IGNORE_UNKNOWN=true ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_log_contains "$WORK/$CASE/out.log" "ignoring hidden model entry"
expect_file "model/1/model.onnx"
expect_absent "model/.DS_Store"

new_case catalog-override-collision-inactive
printf 'onnx-a' >"$STAGE/model-a.onnx"
printf 'onnx-b' >"$STAGE/model.a.onnx"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_file "model-a/1/model.onnx"
expect_file "model.a/1/model.onnx"

new_case catalog-override-collision
printf 'onnx-a' >"$STAGE/model-a.onnx"
printf 'onnx-b' >"$STAGE/model.a.onnx"
TRT_PRECISION_MODEL_A=fp32 ONNX_BACKEND=tensorrt run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "active environment override suffix 'MODEL_A'"
expect_log_contains "$WORK/$CASE/out.log" "rename one model or remove"

new_case explicit-override-collision
printf 'onnx-a' >"$STAGE/a.onnx"
printf 'onnx-b' >"$STAGE/b.onnx"
MODEL_INPUTS="model-a=$STAGE/a.onnx
model.a=$STAGE/b.onnx" TRT_PRECISION_MODEL_A=fp32 ONNX_BACKEND=tensorrt run_stager
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "active environment override suffix 'MODEL_A'"

new_case ensemble-override-suffix-collision
printf 'platform: "ensemble"\nensemble_scheduling {}\n' >"$STAGE/ens-a.pbtxt"
printf 'platform: "ensemble"\nensemble_scheduling {}\n' >"$STAGE/ens_a.pbtxt"
REPOSITORY_LAYOUT=triton run_stager
expect_exit $? 0
expect_file "ens-a/config.pbtxt"
expect_file "ens_a/config.pbtxt"

new_case config-overlay-log
printf 'onnx' >"$STAGE/model.onnx"
printf 'name: "model"\n' >"$STAGE/model.pbtxt"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_log_contains "$WORK/$CASE/out.log" "applied config overlay"
printf 'name: "model"\nmax_batch_size: 8\n' >"$STAGE/model.pbtxt"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_log_contains "$WORK/$CASE/out.log" "applied config overlay"
ONNX_BACKEND=onnx_runtime run_stager
expect_exit $? 0
expect_log_not_contains "$WORK/$CASE/out.log" "applied config overlay"

# =============================================================================
echo
echo "=== Summary ==="
echo "passed: $PASS"
echo "failed: $FAIL"
[ "$FAIL" -eq 0 ]
