# Third-party code

| Path | Origin | License |
|---|---|---|
| `e2esd/ssl/wavlm/` | WavLM ([microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm)), the copy in [s3prl](https://github.com/s3prl/s3prl) commit `7ab62aa`. One change: the in-place `x += x_conv` is now `x = x + x_conv` | MIT, © 2021 Microsoft |
| `e2esd/ssl/wav2vec2/wav2vec2_model.py` | wav2vec 2.0 model code of [fairseq](https://github.com/facebookresearch/fairseq), the copy in s3prl 0.4.17 (`s3prl/upstream/wav2vec2/wav2vec2_model.py`), without changes | MIT, © Facebook, Inc. and its affiliates |
| `e2esd/ssl/upstream.py` | re-implementation of the s3prl upstream interface (hidden-state hooks, checkpoint loading) of s3prl commit `7ab62aa` | MIT (s3prl at `7ab62aa`, WavLM expert © Microsoft) |
| `e2esd/models/backends/_gmlp_block.py` | gMLP of [labml.ai annotated paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) in the version of the PartialSpoof code, without the unused variants | MIT, © 2020 Varuna Jayasiri |
| `e2esd/models/backends/gmlp.py` | frame-level gMLP of [PartialSpoof](https://github.com/nii-yamagishilab/PartialSpoof) (`03multireso`, `MaxPool1dLin_gmlp_scales`) | BSD-3-Clause, © 2023 Lin Zhang, NII |
| `e2esd/models/eda.py` | `EncoderDecoderAttractor` of [BUT EEND](https://github.com/BUTSpeechFIT/EEND) (`eend/backend/models.py`), rewritten. Changes: the time shuffle is inside the module (fixed seed in evaluation), and the module adds a weighted existence BCE | MIT, © 2019 Hitachi, Ltd., © 2022 Brno University of Technology |
| `e2esd/models/backends/conformer.py` | Conformer of [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) (`diarizen/models/module/conformer.py`), without the relative positional encoding | MIT, © 2024 BUT Speech@FIT |
| `e2esd/_vendor/dscore/` | [dscore](https://github.com/nryant/dscore) commit `e02f949`: the `scorelib` package and `LICENSE`, without changes to the code. This repository adds `__init__.py`. It does not include the command-line scripts, their `scorelib/argparse.py` helper, `scorelib/md-eval-22.pl` (for DER only), the tests and the README | BSD-2-Clause, © 2018 Neville Ryant. `scorelib/six.py`: MIT, © 2010-2015 Benjamin Peterson (license text in the file) |
| `e2esd/metrics/eer.py` (`compute_det_curve`, `compute_eer`) | `eval_metrics.py` of the [ASVspoof 2019 t-DCF package](https://www.asvspoof.org/resources/tDCF_python_v2.zip). Changes: the scores are cast to float64, and `compute_eer` is shorter and returns Python floats | **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** (exception to the MIT license of this repository) |

The code downloads these pre-trained SSL weights at their first use. They are the converted
checkpoints of s3prl:

* WavLM-Large (© Microsoft, MIT)
* wav2vec 2.0 Large LV-60 + CV + SWBD + Fisher (© Facebook, MIT)
* XLS-R 300M (© Facebook, MIT)

## License texts

### MIT License

The parts marked MIT above are used under the MIT License with these copyright notices:

```
Copyright (c) Microsoft Corporation                (WavLM)
Copyright (c) Facebook, Inc. and its affiliates    (wav2vec 2.0, fairseq)
Copyright (c) 2021 s3prl development team          (s3prl, commit 7ab62aa)
Copyright (c) 2020 Varuna Jayasiri                 (gMLP, labml.ai)
Copyright (c) 2019 Hitachi, Ltd.                   (EEND-EDA)
Copyright (c) 2022 Brno University of Technology   (BUT EEND)
Copyright (c) 2024 BUT Speech@FIT                  (DiariZen)
```

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### BSD 3-Clause License (PartialSpoof)

```
BSD 3-Clause License

Copyright (c) 2023, Lin Zhang, National Institute of Informatics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
