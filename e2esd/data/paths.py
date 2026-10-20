"""
Directory layout of PartialSpoof. scripts/download_partialspoof.sh and
scripts/prepare_partialspoof.py make this layout:

<db_root>/
  {train,dev,eval}/con_wav/*.wav            PartialSpoof v1.2 audio files (Zenodo 5766198)
  {train,dev,eval}/{split}.lst              utterance lists
  segment_labels/{split}_seglab_0.02.npy    official 20 ms segment labels, bonafide or spoof (EER)
  timestamps_v1.3/vad_20sil/{split}/*.vad   v1.3 label files (detailed timestamps) for training
  timestamps_v1.3/vad_20sil_noconp/...      ConP changed to non-speech (ConP ablation)
  rttms/{dev,eval}/ref_spoof_dia.rttm       reference RTTMs for scoring
"""

import os


def split_paths(db_root, split, vad_dir='vad_20sil'):
    return {
        'wav': os.path.join(db_root, split, 'con_wav'),
        'list': os.path.join(db_root, split, f'{split}.lst'),
        'vad': os.path.join(db_root, 'timestamps_v1.3', vad_dir, split),
        'seglab': os.path.join(db_root, 'segment_labels', f'{split}_seglab_0.02.npy'),
        'ref_rttm': os.path.join(db_root, 'rttms', split, 'ref_spoof_dia.rttm'),
    }
