"""
Audio and list files, and the PartialSpoof dataset for training. Each dataset item is a
full-length waveform with the class of each frame.
"""

import os

import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .labels import BONAFIDE_ID, binary_labels, count_classes, frame_labels

# Inference and detection cut longer utterances to this length (s). The gMLP back-end accepts
# at most 2001 frames (40 s).
MAX_SECONDS = 39.5


def read_list(path):
    """The first column of each line that is not empty."""
    with open(path) as f:
        return [line.split()[0] for line in f if line.strip()]


def read_audio(path, sample_rate=16000):
    """(samples,) float32 waveform of the first channel. The function resamples it to
    sample_rate if necessary."""
    wav, sr = sf.read(path, dtype='float32', always_2d=True)
    wav = torch.from_numpy(wav[:, 0])
    return wav if sr == sample_rate else torchaudio.functional.resample(wav, sr, sample_rate)


def load_wav(wav_dir, utt, sample_rate=16000):
    """The waveform of <wav_dir>/<utt>.wav or .flac (see read_audio)."""
    for ext in ('.wav', '.flac'):
        path = os.path.join(wav_dir, utt + ext)
        if os.path.exists(path):
            return read_audio(path, sample_rate)
    raise FileNotFoundError(f'{utt}(.wav|.flac) not found in {wav_dir}')


class PartialSpoofDataset(Dataset):
    """
    Each item is a dict with these keys:
      wav         (samples,) waveform
      labels      (T,) class of each 20 ms frame (-1 = no target)
      n_classes   K (>= 1)
      has_bona    True if class 0 is bonafide
      utt         utterance id
      bin_labels  (T,) bonafide / spoof target of each frame, only if with_bin_labels=True
    """

    def __init__(self, wav_dir, vad_dir, file_list, frame_shift=0.02, sample_rate=16000,
                 nonspeech='ignore', with_bin_labels=False):
        self.wav_dir, self.vad_dir = wav_dir, vad_dir
        self.frame_shift, self.sample_rate = frame_shift, sample_rate
        self.nonspeech = nonspeech
        self.with_bin_labels = with_bin_labels
        self.utts = read_list(file_list)
        # K of each utterance (items, balanced sampler and training log)
        self.n_classes = [count_classes(self._vad(u)) for u in self.utts]

    def _vad(self, utt):
        return os.path.join(self.vad_dir, utt + '.vad')

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = self.utts[idx]
        labels, class_ids = frame_labels(self._vad(utt), self.frame_shift, self.nonspeech)
        item = {'wav': load_wav(self.wav_dir, utt, self.sample_rate),
                'labels': torch.from_numpy(labels),
                'n_classes': self.n_classes[idx],
                'has_bona': BONAFIDE_ID in class_ids,
                'utt': utt}
        if self.with_bin_labels:
            item['bin_labels'] = torch.from_numpy(binary_labels(self._vad(utt), self.frame_shift))
        return item


def _pad(seqs, value):
    return pad_sequence(seqs, batch_first=True, padding_value=value)


def collate_fn(items):
    """Pads the waveforms with zeros and the labels with -1 (no target)."""
    return {'wav': _pad([it['wav'] for it in items], 0.0),
            'labels': _pad([it['labels'] for it in items], -1),
            'n_classes': [it['n_classes'] for it in items],
            'has_bona': [it['has_bona'] for it in items],
            'utt': [it['utt'] for it in items],
            'bin_labels': _pad([it['bin_labels'] for it in items], -1)
            if 'bin_labels' in items[0] else None}
