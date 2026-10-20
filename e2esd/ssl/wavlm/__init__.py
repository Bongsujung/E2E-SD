# This package is a copy from s3prl (commit 7ab62aa). The origin is WavLM of microsoft/unilm
# (MIT License). The only change from upstream: the in-place `x += x_conv` is now
# `x = x + x_conv` (safe for autograd).
