<div align="center">

<h2>[NeurIPS 2026] Clustering-Free End-to-End Spoof Diarization<br>with Encoder-Decoder Attractors</h2>

Bongsu Jung<sup>1</sup> &nbsp;&nbsp; Donghee Kim<sup>2</sup> &nbsp;&nbsp; Wooil Kim<sup>1</sup>

<sup>1</sup>Department of Computer Science and Engineering &nbsp;&nbsp;
<sup>2</sup>Department of Artificial Intelligence<br>
Incheon National University, Republic of Korea

<img src="https://img.shields.io/badge/NeurIPS-2026-brightgreen">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>

</div>

<p align="center"><img src="assets/overview.png" width="95%"></p>

This repository contains the official PyTorch code of **Clustering-Free End-to-End Spoof
Diarization with Encoder-Decoder Attractors** (NeurIPS 2026).

Spoof diarization finds the spoofed regions of a partially spoofed utterance. It also groups
these regions by the spoofing method that made them. E2E-SD does these two tasks with one
network and without clustering.

* **No clustering:** A WavLM-Large front-end and a gMLP back-end give the frame embeddings. An
  LSTM encoder-decoder makes one attractor for each class. The model assigns each frame to its
  most similar attractor.
* **Estimated number of classes:** Each attractor has an existence probability. Thus, inference
  does not need the oracle number of classes.
* **Bonafide attractor:** Training fixes the first attractor a_0 to bonafide. Thus, the score
  cos(e_t, a_0) of each frame also localizes and detects spoofing.

## Results

The table shows the results on the PartialSpoof evaluation set with the estimated K. Each value
is the mean ± std over three runs (%). Lower is better, except for K-acc.

| Front-end | JI | JER | Known | Varied | Unknown | K-acc |
|---|---|---|---|---|---|---|
| WavLM-Large (default) | 8.32&nbsp;±&nbsp;0.40 | 16.71&nbsp;±&nbsp;0.29 | 9.61&nbsp;±&nbsp;0.35 | 8.80&nbsp;±&nbsp;0.33 | 23.58&nbsp;±&nbsp;0.50 | 87.42&nbsp;±&nbsp;0.11 |
| XLS-R 300M | 10.77&nbsp;±&nbsp;1.74 | 19.60&nbsp;±&nbsp;1.60 | 7.94&nbsp;±&nbsp;0.19 | 10.18&nbsp;±&nbsp;0.70 | 28.69&nbsp;±&nbsp;2.75 | 82.27&nbsp;±&nbsp;2.44 |
| wav2vec 2.0 Large (3C front-end) | 11.56&nbsp;±&nbsp;0.39 | 21.90&nbsp;±&nbsp;0.56 | 12.26&nbsp;±&nbsp;0.68 | 12.77&nbsp;±&nbsp;0.66 | 30.25&nbsp;±&nbsp;0.47 | 84.53&nbsp;±&nbsp;1.20 |

## Installation

```bash
git clone https://github.com/Bongsujung/E2E-SD.git && cd E2E-SD
conda create -n e2esd python=3.10 -y && conda activate e2esd
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt -e .
python -m pytest -q          # unit tests (no data, GPU or download needed)
```

We tested the code with Python 3.10, PyTorch 2.1.2 and CUDA 11.8 on one NVIDIA RTX 4090.
`requirements.txt` gives the exact package versions that we used for the paper. This repository
includes the model code of the SSL front-ends (WavLM, wav2vec 2.0). Thus, you do not need s3prl
or fairseq. When you use an SSL front-end for the first time, the code downloads its pre-trained
weights to `$E2ESD_CACHE/ssl` (default `~/.cache/e2esd/ssl`).

## Data

E2E-SD uses these parts of [PartialSpoof](https://github.com/nii-yamagishilab/PartialSpoof):

* the v1.2 audio and the official 20 ms segment labels (Zenodo
  [5766198](https://zenodo.org/records/5766198))
* the v1.3 detailed timestamps, with the labels bonafide, A01-A19, concatenated part (ConP) and
  non-speech. The Zenodo page has the Google Drive link to these timestamps.

```bash
bash scripts/download_partialspoof.sh data/PartialSpoof
# or, with the database already on disk:
python scripts/prepare_partialspoof.py --db-root /path/to/PartialSpoof/database [--noconp]
```

`prepare_partialspoof.py` checks the directory layout below. Then it writes the reference RTTMs
for scoring. With `--noconp`, it also writes the label files for the ConP ablation.

```
database/
├── {train,dev,eval}/con_wav/*.wav, {split}.lst
├── segment_labels/{split}_seglab_0.02.npy      official 20 ms labels (EER)
├── timestamps_v1.3/vad_20sil/{split}/*.vad     training labels
├── timestamps_v1.3/vad_20sil_noconp/...        ConP ablation (--noconp)
└── rttms/{dev,eval}/ref_spoof_dia.rttm         reference for scoring
```

`eval.lst` lists 71,239 utterances. Two of them, `CON_E_0034982` and `CON_E_0058039`, have no
audio file and no label file in PartialSpoof v1.2. All scripts skip these two utterances. Thus,
the scripts score 71,237 utterances of the evaluation set.

## Training and evaluation

```bash
DB=/path/to/PartialSpoof/database
bash run.sh configs/e2esd.yaml exp/e2esd/s777 $DB 777        # train, infer on eval, score
```

`run.sh` runs these three steps:

```bash
python scripts/train.py --config configs/e2esd.yaml --out-dir exp/e2esd/s777 \
    --set data.db_root=$DB training.seed=777
python scripts/infer.py --checkpoint exp/e2esd/s777/best.pt --db-root $DB --split eval \
    --out-dir exp/e2esd/s777/eval
python scripts/evaluate.py --infer-dir exp/e2esd/s777/eval --db-root $DB --split eval
```

To get the mean ± std over runs, use `python scripts/summarize.py exp/e2esd/s* --sub eval`.

One run takes 5-7 h on one NVIDIA RTX 4090 and uses approximately 17 GB of GPU memory. Inference
on the evaluation set takes less than one hour. You can change any config key with the `--set`
option of `scripts/train.py`. If the key does not exist, the script stops with an error.

`configs/e2esd.yaml` is the config of the default model. All other configs start from it and
change one setting:

| Config | Experiment in the paper |
|---|---|
| `frontend/xlsr_300m.yaml`, `frontend/w2v2_large.yaml` | front-end controls (main results) |
| `ablation/loss_ce.yaml`, `loss_p2sgrad.yaml`, `loss_contrast.yaml` | assignment loss |
| `ablation/bk_transformer.yaml`, `bk_conformer.yaml` | back-end |
| `ablation/ssl_frozen.yaml` | SSL front-end frozen |
| `ablation/head_loc.yaml`, `loc_bin.yaml` | localization supervision (separate head, L_bin) |
| `ablation/match_free.yaml` | bonafide anchoring (free matching) |
| `ablation/ex_wbce.yaml`, `ex_balanced.yaml` | existence training |
| `ablation/eda_noshuffle.yaml` | EDA time shuffle |
| `ablation/lab_nsp_inherit.yaml` | non-speech frames in training |
| `ablation/lab_noconp.yaml` | concatenated-part (ConP) class |

The ablations in the paper use the seeds 777, 1234 and 2025. For oracle K, use
`scripts/infer.py --oracle-k`. One inference pass can give the results for different values of
the existence threshold θ and K_max, and for oracle K. Run these commands:

```bash
python scripts/infer.py --checkpoint exp/e2esd/s777/best.pt --db-root $DB --save-scores 11 \
    --out-dir exp/e2esd/s777/eval_scores
python scripts/sweep_k.py --infer-dir exp/e2esd/s777/eval_scores --db-root $DB
```

## Transfer and detection

The first three commands below build the test sets outside PartialSpoof from their official
releases. The last command runs the transfer and detection tests:

```bash
python scripts/build_llamaps.py --llamaps-dir /data/LlamaPartialSpoof --out-dir exp/sets
python scripts/build_partialedit.py --pe-dir /data/PartialEdit_v1.1 --out-dir exp/sets
python scripts/build_detect_lists.py --out-dir exp/sets/detect --asv2019-la /data/LA \
    --asv2021-la /data/ASVspoof2021_LA_eval --asv2021-df /data/ASVspoof2021_DF_eval \
    --asv2021-keys /data/keys --itw /data/release_in_the_wild
bash scripts/run_transfer.sh exp/e2esd/s777 exp/sets
```

* [LlamaPartialSpoof](https://github.com/hieuthi/LlamaPartialSpoof): The utterance id contains
  the name of the generator. Each spoof segment gets this generator as its label. Thus, a
  partially spoofed utterance has K = 2. The EERs use all 76,228 utterances of part a. JER, JI
  and K-acc use only the 32,194 partially spoofed utterances of part a. Part b gives two
  separate sets, cut-and-paste and overlap-add.
* [PartialEdit](https://doi.org/10.5281/zenodo.18829689) v1.1: The set contains the 53,727
  utterances of the official evaluation speakers. The editors are VoiceCraft and SSR-Speech.
  Each utterance has K = 2. First, extract E1.tar.gz, E2.tar.gz and PartialEdit_spk.zip (see
  `scripts/build_partialedit.py`).
* ASVspoof 2019 LA, ASVspoof 2021 LA, ASVspoof 2021 DF and In-the-Wild: The detection lists
  contain every trial of each evaluation set.

On the external sets, the scoring uses no voice activity detection. For the EERs on
LlamaPartialSpoof and PartialEdit, a 20 ms frame is spoof if one or more spoof segments
overlap it.

### Check the test sets

The scripts build the same reference files that we used for the paper. To compare, use
`sha256sum <file>` and the first 16 characters below.

| File | Utterances | SHA-256 (first 16 characters) |
|---|---|---|
| `$DB/rttms/dev/ref_spoof_dia.rttm` | 24,844 | `23ee4a61b803b24f` |
| `$DB/rttms/eval/ref_spoof_dia.rttm` | 71,237 | `be7fed602f6e5297` |
| `exp/sets/llamaps_a/ref_spoof_dia.rttm` | 76,228 | `c2d934459a82cf63` |
| `exp/sets/llamaps_cf/ref_spoof_dia.rttm` | 32,194 | `ca68449e91f6df3a` |
| `exp/sets/llamaps_cp/ref_spoof_dia.rttm` | 32,194 | `f11eea8734874ead` |
| `exp/sets/llamaps_oa/ref_spoof_dia.rttm` | 32,194 | `266198e9e9e4b05f` |
| `exp/sets/partialedit/ref_spoof_dia.rttm` | 53,727 | `bf1b94c3ea09f360` |

The detection lists contain the audio paths of your computer, so they have no fixed checksum.
They contain 71,237 (2019 LA), 181,566 (2021 LA), 611,829 (2021 DF) and 31,779 (In-the-Wild)
trials.

## Pre-trained models

We will release the trained models in this repository.

## Metrics

`e2esd/metrics/jer.py` computes the metrics of Zhang et al. (Interspeech 2024, Eq. 1-4, and
arXiv version, App. A.1). These metrics adapt the Jaccard error rate of
[dscore](https://github.com/nryant/dscore):

* The Hungarian algorithm maps each reference class to a hypothesis cluster. The error of the
  class is J = (false alarm + miss) / |reference ∪ hypothesis|. A reference class that has no
  cluster has J = 1.
* **JER** is the mean J over all (utterance, spoofing method) pairs. The Known, Varied and
  Unknown JERs pool the J values of these spoofing methods:
  * Known: A16, A19
  * Varied: A07, A08, A09, A17
  * Unknown: A10-A15, A18
* **JI** is the mean J of the bonafide class over the utterances that contain bonafide speech.
* The scoring uses only the reference speech. Thus, it ignores non-speech and ConP.

dscore is the scorer. The J of each class comes from the frame conversion and the contingency
matrix of dscore at a 0.1 ms step. This repository includes this dscore code without changes.
You cannot use the default dscore step of 10 ms, because some reference segments are shorter
than 10 ms.

The EERs use the bonafide score cos(e_t, a_0) of each frame. On PartialSpoof, the EERs follow
the official protocol:

* The seg-EER uses the official 20 ms segment labels.
* The utt-EER uses the minimum frame score as the utterance score.

## Notes

* Each run sets a seed. But the cuDNN LSTM and scatter kernels are not deterministic. Thus, two
  runs with the same seed give slightly different results. With `--deterministic`,
  `scripts/train.py` uses deterministic algorithms where PyTorch has them, but it is slower.
* Training uses full-length utterances. In a mini-batch, zero padding makes all utterances the
  same length. As in the paper runs, the network does not mask the padding. Only the losses
  ignore the padded frames.
* The gMLP back-end accepts at most 2001 frames (40 s). Thus, inference and detection cut
  longer utterances at 39.5 s.

## Repository layout

```
e2esd/
├── models/
│   ├── model.py         E2E-SD: training loss and diarize() for inference
│   ├── encoder.py       SSL front-end, weighted sum of its layers, back-end
│   ├── eda.py           encoder-decoder attractors and existence probabilities
│   └── backends/        gMLP (default), Transformer, Conformer
├── ssl/                 WavLM and wav2vec 2.0 model code, weight download
├── losses.py            assignment losses and class-to-attractor matching
├── data/                PartialSpoof dataset, labels, balanced sampler
├── metrics/             JER and JI (dscore), EER, K accuracy
├── utils/               config, RTTM output, logging, seed, checkpoint file loading
├── checkpoint.py        load_model()
└── _vendor/dscore/      dscore scoring library (unchanged)
configs/                 e2esd.yaml (default), frontend/, ablation/
scripts/                 data preparation, train, infer, evaluate, transfer and detection
tests/                   unit tests
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{jung2026e2esd,
  title     = {Clustering-Free End-to-End Spoof Diarization with Encoder-Decoder Attractors},
  author    = {Jung, Bongsu and Kim, Donghee and Kim, Wooil},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

A KOITA grant funded by MSIT (KOITA20250002-38) supported this work.

The code uses or adapts code from these projects. `THIRD_PARTY_NOTICES.md` gives the details and
the licenses.

* EDA: [BUT EEND](https://github.com/BUTSpeechFIT/EEND) and
  [Hitachi EEND](https://github.com/hitachi-speech/EEND)
* gMLP back-end and EER protocol: [PartialSpoof](https://github.com/nii-yamagishilab/PartialSpoof)
  and [labml.ai](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
* Conformer back-end: [DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
* SSL front-ends: [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm),
  [fairseq](https://github.com/facebookresearch/fairseq) and [s3prl](https://github.com/s3prl/s3prl)
* Scoring: [dscore](https://github.com/nryant/dscore) and the ASVspoof 2019 t-DCF package

## License

This repository is under the MIT License (`LICENSE`). Code from other projects keeps its license
(see `THIRD_PARTY_NOTICES.md`). Note one exception: `compute_det_curve` and `compute_eer` in
`e2esd/metrics/eer.py` come from the ASVspoof 2019 t-DCF package. These two functions are under
the CC BY-NC-SA 4.0 license, not under the MIT License. The datasets have their own licenses.
