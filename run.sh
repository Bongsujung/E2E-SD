#!/usr/bin/env bash
# Train the model with one config and one seed (one run). Then run inference on the PartialSpoof
# eval split and score the result.
#
#   bash run.sh <config> <exp_dir> <db_root> [seed] [stage]
#
# [seed] is 777 by default. [stage] is the first step (default 1):
#   1  training, inference and scoring
#   2  inference and scoring
#   3  scoring only
#
# Examples:
#  bash run.sh configs/e2esd.yaml exp/e2esd/s777 /data/PartialSpoof/database 777
#  for s in 777 1234 2025; do bash run.sh configs/ablation/loss_ce.yaml exp/loss_ce/s$s $DB $s; done
set -euo pipefail
cd "$(dirname "$0")"
config=${1:?usage: run.sh <config> <exp_dir> <db_root> [seed] [stage]}
exp=${2:?exp_dir}; db=${3:?db_root}; seed=${4:-777}; stage=${5:-1}

if [ "${stage}" -le 1 ]; then
    python scripts/train.py --config "${config}" --out-dir "${exp}" \
        --set data.db_root="${db}" training.seed="${seed}"
fi
if [ "${stage}" -le 2 ]; then
    python scripts/infer.py --checkpoint "${exp}/best.pt" --db-root "${db}" --split eval \
        --out-dir "${exp}/eval"
fi
python scripts/evaluate.py --infer-dir "${exp}/eval" --db-root "${db}" --split eval
