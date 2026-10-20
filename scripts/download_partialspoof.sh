#!/usr/bin/env bash
# Download PartialSpoof v1.2 (Zenodo 5766198) and the v1.3 detailed timestamps. The Zenodo page
# has the link to the Google Drive folder of the timestamps. Then check the directory layout and
# make the reference RTTMs.
#
#   bash scripts/download_partialspoof.sh data/PartialSpoof    # makes data/PartialSpoof/database
set -euo pipefail
target=${1:-data/PartialSpoof}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${target}" && cd "${target}"

for f in train dev eval segment_labels_v1.2; do
    if [ ! -f ".done_${f}" ]; then
        wget -c "https://zenodo.org/records/5766198/files/database_${f}.tar.gz?download=1" \
            -O "database_${f}.tar.gz"
        tar -xzf "database_${f}.tar.gz" && rm "database_${f}.tar.gz" && touch ".done_${f}"
    fi
done

# v1.3 timestamps: vad_20sil.tar.gz, and label2num_all with the label ids of bonafide, A01-A19,
# ConP and non-speech
if [ ! -d database/timestamps_v1.3/vad_20sil ]; then
    python -m gdown --folder \
        "https://drive.google.com/drive/folders/1kKW3GBuooPkAl64Zyv6WPgICH5LZtnQR" \
        -O database/timestamps_v1.3
    tar -xzf database/timestamps_v1.3/vad_20sil.tar.gz -C database/timestamps_v1.3
fi
python "${here}/prepare_partialspoof.py" --db-root "$(pwd)/database"
