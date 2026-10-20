#!/usr/bin/env bash
# Run the transfer and detection tests for one run (paper Section 4.5, App. E).
#
#   bash scripts/run_transfer.sh <run_dir> <sets_dir>
#
# <sets_dir> contains the outputs of these scripts:
#   scripts/build_llamaps.py
#   scripts/build_partialedit.py
#   scripts/build_detect_lists.py (with --out-dir <sets_dir>/detect)
# The script skips each set that is not in <sets_dir>.
# Results: <run_dir>/<set>/results.json and <run_dir>/detect_<set>.tsv.
set -euo pipefail
cd "$(dirname "$0")/.."
run=${1:?usage: run_transfer.sh <run_dir> <sets_dir>}; sets=${2:?sets_dir}
ckpt=${run}/best.pt

for name in llamaps_a llamaps_cp llamaps_oa partialedit; do
    set_dir=${sets}/${name}; out=${run}/${name}
    [ -d "${set_dir}" ] || { echo "[transfer] no ${set_dir}, skipping"; continue; }
    python scripts/infer.py --checkpoint "${ckpt}" --set-dir "${set_dir}" --out-dir "${out}"
    # LlamaPartialSpoof part a: the EERs use all utterances. JER, JI and K use only the
    # partially spoofed utterances.
    ref=${set_dir}/ref_spoof_dia.rttm
    [ "${name}" = llamaps_a ] && ref=${sets}/llamaps_cf/ref_spoof_dia.rttm
    python scripts/evaluate_external.py --infer-dir "${out}" --set-dir "${set_dir}" \
        --jer-ref "${ref}"
done

for name in 19la 21la 21df itw; do
    list=${sets}/detect/${name}.tsv
    [ -f "${list}" ] || { echo "[detect] no ${list}, skipping"; continue; }
    python scripts/detect.py --checkpoint "${ckpt}" --list "${list}" \
        --out "${run}/detect_${name}.tsv"
done
