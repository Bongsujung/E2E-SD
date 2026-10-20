"""
Training log. The logger writes to:
  * the console and <out>/log_train.txt
  * <out>/metrics.csv (one row for each epoch)
  * <out>/config.json (the config of the run)
  * TensorBoard, if it is installed
"""

import json
import os
import time
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def _fmt(terms):
    return ' '.join(f'{k}={v:.5f}' for k, v in terms.items())


class TrainLogger:
    def __init__(self, out_dir, config):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'metrics.csv')
        self._header_written = os.path.isfile(self.csv_path)   # resume: keep the old header
        self.log_file = open(os.path.join(out_dir, 'log_train.txt'), 'a')
        self.tb = SummaryWriter(os.path.join(out_dir, 'tb')) if SummaryWriter else None
        self.start = time.time()
        self._epoch_start = None
        self._step = 0
        with open(os.path.join(out_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        self.log(f'=== E2E-SD training started {datetime.now():%Y-%m-%d %H:%M:%S} ===')

    def log(self, msg):
        print(msg, flush=True)
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def begin_epoch(self):
        self._epoch_start = time.time()

    def log_step(self, terms):
        self._step += 1
        if self.tb:
            for k, v in terms.items():
                self.tb.add_scalar(f'step/{k}', v, self._step)

    def log_epoch(self, epoch, n_epochs, train, dev, lr, is_best):
        elapsed = time.time() - self._epoch_start
        self.log(f'Epoch {epoch}/{n_epochs} | train: {_fmt(train)} | dev: {_fmt(dev)} | '
                 f'lr={lr:.2e} | {elapsed:.0f}s' + (' | best' if is_best else ''))
        if self.tb:
            for k, v in train.items():
                self.tb.add_scalar(f'train/{k}', v, epoch)
            for k, v in dev.items():
                self.tb.add_scalar(f'dev/{k}', v, epoch)
            self.tb.add_scalar('lr', lr, epoch)
            self.tb.flush()
        row = {'epoch': epoch, 'lr': lr, 'time_s': elapsed,
               **{f'train_{k}': v for k, v in train.items()},
               **{f'dev_{k}': v for k, v in dev.items()}, 'is_best': int(is_best)}
        with open(self.csv_path, 'a') as f:
            if not self._header_written:
                f.write(','.join(row) + '\n')
                self._header_written = True
            f.write(','.join(str(v) for v in row.values()) + '\n')

    def close(self):
        self.log(f'=== training finished after {(time.time() - self.start) / 3600:.1f} h ===')
        self.log_file.close()
        if self.tb:
            self.tb.close()
