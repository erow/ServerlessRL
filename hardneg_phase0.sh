#!/usr/bin/env bash
# Phase 0 of the hard-negative inpainting study: shortcut probes (CPU) and base-model accuracy per distractor scheme.
# Run from the parent project root with the serverless venv active and PYTHONPATH set (as in run.sh):
#
#   COCO_VAL=.../val2017.zip INSTANCES=.../instances_val2017.json [STUFF=.../stuff_val2017.json] \
#   OUT=$SERVERLESS_OUTPUT/hardneg/phase0 bash serverless/hardneg_phase0.sh [index|probe|vlm|all]
#
# The VLM step calls W&B Inference (WANDB_API_KEY) with Qwen3.8-27B, thinking on, 16k tokens, 8 samples per puzzle.
set -euo pipefail
: "${COCO_VAL:?path to val2017 directory or ZIP}" "${OUT:?output directory}"
INDEX="${INDEX:-$OUT/val2017_labels.npz}"
N_PROBE="${N_PROBE:-500}" N_VLM="${N_VLM:-200}" SAMPLES="${SAMPLES:-8}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}" MAX_TOKENS="${MAX_TOKENS:-16384}" CONC="${CONC:-16}"
mkdir -p "$OUT"
step="${1:-all}"

ORACLE="\"oracle_index\": \"$INDEX\""
declare -A SCHEMES=(
  [ssl-r1]='{}'
  [spatial]='{"mask": "square", "distractors": "spatial"}'
  [spatial-uniform]='{"mask": "square", "distractors": "spatial", "uniform_resample": true}'
  [random]='{"mask": "square", "distractors": "random"}'
  [random-nofilter]='{"mask": "square", "distractors": "random", "boundary_filter": null}'
  [random-color]='{"mask": "square", "distractors": "random", "color_match": true}'
  [oracle]="{\"mask\": \"square\", \"distractors\": \"oracle-cat,oracle-dist,oracle-dist\", $ORACLE}"
  [oracle-nofilter]="{\"mask\": \"square\", \"distractors\": \"oracle-cat,oracle-dist,oracle-dist\", \"boundary_filter\": null, $ORACLE}"
  [oracle-color]="{\"mask\": \"square\", \"distractors\": \"oracle-cat,oracle-dist,oracle-dist\", \"color_match\": true, $ORACLE}"
  [mixed]="{\"mask\": \"square\", \"distractors\": \"rot90,oracle-cat,oracle-dist\", $ORACLE}"
)
VLM_SCHEMES=(spatial ssl-r1 random oracle oracle-color mixed)
BLIND_SCHEMES=(spatial random oracle)

if [[ $step == index || $step == all ]] && [[ ! -f $INDEX ]]; then
  : "${INSTANCES:?instances_val2017.json}"
  python -m serverless.hardneg build-index --instances "$INSTANCES" ${STUFF:+--stuff "$STUFF"} --out "$INDEX"
fi

if [[ $step == probe || $step == all ]]; then
  args=()
  for k in "${!SCHEMES[@]}"; do args+=(--scheme "$k=${SCHEMES[$k]}"); done
  python -m serverless.hardneg probe --images "$COCO_VAL" --n "$N_PROBE" --out-dir "$OUT/probe" "${args[@]}"
fi

if [[ $step == vlm || $step == all ]]; then
  run() {  # name, params JSON, samples
    python -m serverless.difficulty --base-url https://api.inference.wandb.ai/v1 --api-key "$WANDB_API_KEY" \
      --model "$MODEL" --tasks inpainting --task-params "{\"inpainting\": $2}" --images "$COCO_VAL" \
      --n-per-task "$N_VLM" --samples "$3" --temperature 1.0 --enable-thinking \
      --max-completion-tokens "$MAX_TOKENS" --max-concurrency "$CONC" --no-stream \
      --output-dir "$OUT/vlm/$1"
  }
  for k in "${VLM_SCHEMES[@]}"; do run "$k" "${SCHEMES[$k]}" "$SAMPLES"; done
  for k in "${BLIND_SCHEMES[@]}"; do
    run "$k-blind" "$(python -c 'import json,sys; d=json.loads(sys.argv[1]); d["blind"]=True; print(json.dumps(d))' \
      "${SCHEMES[$k]}")" 4
  done
fi
