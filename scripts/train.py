#!/usr/bin/env python
"""
Train E2E-SD on PartialSpoof.

    python scripts/train.py --config configs/e2esd.yaml --out-dir exp/e2esd/s777 \
        --set data.db_root=/path/to/PartialSpoof/database training.seed=777

Output files in <out-dir>:
  best.pt        checkpoint with the lowest dev loss
  latest.pt      checkpoint of the last epoch. An interrupted run starts again from it.
  log_train.txt  training log
  metrics.csv    losses of each epoch
  config.json    config of the run
  tb/            TensorBoard logs (only if TensorBoard is installed)
Training stops when the dev loss does not improve for training.patience epochs.
"""

import argparse
import os
import sys
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2esd.data import (BalancedSpoofSampler, PartialSpoofDataset, collate_fn,  # noqa: E402
                        split_paths)
from e2esd.models import E2ESD  # noqa: E402
from e2esd.utils import (load_config, parse_overrides, seed_everything, to_plain,  # noqa: E402
                         torch_load)
from e2esd.utils.logger import TrainLogger  # noqa: E402

TERMS = ('total', 'assign', 'exist', 'loc', 'bin')


def make_dataset(cfg, split):
    """The config key data.nonspeech changes only the labels of the train split. The dev loss
    selects best.pt and always uses the default labels: non-speech frames have no target."""
    dc = cfg.data
    p = split_paths(dc.db_root, split, dc.vad_dir)
    return PartialSpoofDataset(p['wav'], p['vad'], p['list'], dc.frame_shift, dc.sample_rate,
                               nonspeech=dc.nonspeech if split == 'train' else 'ignore',
                               with_bin_labels=cfg.loss.get('lambda_bin', 0.0) > 0)


def run_model(model, batch, device):
    bin_labels = batch['bin_labels']
    return model(batch['wav'].to(device), batch['labels'].to(device), batch['n_classes'],
                 batch['has_bona'], None if bin_labels is None else bin_labels.to(device))


def mean_terms(rows):
    """Mean of each loss term over the utterances. rows: [(terms, batch size), ...]."""
    n = sum(bs for _, bs in rows)
    return {k: sum(t[k] * bs for t, bs in rows) / max(n, 1) for k in TERMS}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc='  dev', ncols=100, leave=False):
        _, terms = run_model(model, batch, device)
        rows.append((terms, len(batch['utt'])))
    return mean_terms(rows)


def main():
    ap = argparse.ArgumentParser(description='Train E2E-SD on PartialSpoof.')
    ap.add_argument('--config', required=True, help='config file, for example configs/e2esd.yaml')
    ap.add_argument('--out-dir', required=True, help='output directory')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='config keys to change, for example training.seed=1234')
    ap.add_argument('--deterministic', action='store_true',
                    help='use deterministic algorithms where PyTorch has them (slower)')
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    tc = cfg.training
    seed_everything(tc.seed, deterministic=args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = TrainLogger(args.out_dir, config=to_plain(cfg))

    train_ds, dev_ds = make_dataset(cfg, 'train'), make_dataset(cfg, 'dev')
    sampler = BalancedSpoofSampler(train_ds, tc.balanced_ratio) if tc.balanced_sampling else None
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=sampler is None,
                              sampler=sampler, collate_fn=collate_fn, num_workers=tc.num_workers,
                              pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=tc.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=tc.num_workers, pin_memory=True)
    logger.log(f'device {device} | train {len(train_ds)} utterances, dev {len(dev_ds)} | '
               f'train K: {dict(sorted(Counter(train_ds.n_classes).items()))}')

    model = E2ESD(cfg).to(device)
    logger.log(f'{model.count_parameters():,} parameters | bona_match={model.bona_match}')

    # The scale (w, b) of the assignment loss and of L_bin uses 0.01x the learning rate and no
    # weight decay.
    scale, other = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (scale if name.startswith(('assign_loss_fn.', 'bin_scale.')) else other).append(p)
    groups = [{'params': other, 'lr': tc.lr}]
    if scale:
        groups.append({'params': scale, 'lr': tc.lr * 0.01, 'weight_decay': 0.0})
    optimizer = torch.optim.Adam(groups, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, tc.lr_decay_step, tc.lr_decay_gamma)

    start, best, bad = 0, float('inf'), 0
    latest = os.path.join(args.out_dir, 'latest.pt')
    if os.path.isfile(latest):
        ckpt = torch_load(latest, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start, best, bad = ckpt['epoch'], ckpt['best_dev_loss'], ckpt['epochs_without_gain']
        logger.log(f'resumed from {latest}: {start} epochs done, best dev loss {best:.6f}')

    done = start
    for epoch in range(start + 1, tc.epochs + 1):
        if bad >= tc.patience:
            break
        logger.begin_epoch()
        model.train()
        rows = []
        optimizer.zero_grad()
        steps = len(train_loader)
        for step, batch in enumerate(tqdm(train_loader, desc=f'epoch {epoch}', ncols=100)):
            loss, terms = run_model(model, batch, device)
            (loss / tc.grad_accum).backward()
            if (step + 1) % tc.grad_accum == 0 or step + 1 == steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            rows.append((terms, len(batch['utt'])))
            logger.log_step(terms)

        train_m = mean_terms(rows)
        dev_m = validate(model, dev_loader, device)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        is_best = dev_m['total'] < best
        best, bad = (dev_m['total'], 0) if is_best else (best, bad + 1)
        logger.log_epoch(epoch, tc.epochs, train_m, dev_m, lr, is_best)

        state = {'model': model.state_dict(), 'config': to_plain(cfg),
                 'ssl_arch_cfg': model.encoder.ssl.arch_cfg, 'epoch': epoch}
        if is_best:
            torch.save({**state, 'dev_loss': best}, os.path.join(args.out_dir, 'best.pt'))
        torch.save({**state, 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'best_dev_loss': best,
                    'epochs_without_gain': bad}, latest)
        done = epoch

    logger.log(f'stopped after {done} epochs; best dev loss {best:.6f}')
    logger.close()


if __name__ == '__main__':
    main()
