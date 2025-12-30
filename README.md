# README

[![Python](https://img.shields.io/badge/python-3.10-blue)](#)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Codacy - Coverage](https://app.codacy.com/project/badge/Coverage/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/ComDec/unirna_tf/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CodeCov - Coverage](https://codecov.io/gh/ComDec/unirna_tf/graph/badge.svg)](https://codecov.io/gh/ComDec/unirna_tf)

The light version of Uni-RNA, which is designed to be more efficient and easier to use.

## Installation

If you unzip the code from compressed file, please run `git init` to initialize the git repository. We need git info. to track the version of the code.

```bash
conda create -n unirna python=3.10
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

## Testing & Coverage

Run tests and generate coverage reports:

```bash
/data1/xw3763/miniforge3/envs/unirna/bin/python -m pytest \
  --cov=unirna_tf \
  --cov-report=term-missing \
  --cov-report=html \
  --cov-report=xml
```

- Terminal summary includes missing lines (`term-missing`)
- HTML report: `htmlcov/index.html`
- XML report: `coverage.xml`

Coverage badge is powered by Codecov (see CI workflow).

## How to use

We provide jupyter notebook to demonstrate how to use the pretrained model. You can find the notebook in the `examples` directory.

For model weights, please download from [Google Drive](https://drive.google.com/file/d/1zzxQa4LHCOHR9GS4MQJ4uFNtMgJsPbuv/view?usp=drive_link) and copy to the root directory of the project, then run:
`tar -zxvf weights.tar.gz`. You will find the model weights is stored in the `weights` directory.

### Quick Start

**!!! You must convert string to uppercase before inputting the sequence to the model !!!**

Sequence "AUcg" is different from "AUCG", all the lowercase letters will be merged and converted to `unk_token` in the tokenizer.

```python
import unirna_tf
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./weights/unirna_L16")
model = AutoModel.from_pretrained("./weights/unirna_L16")

seq = "AUCGGUGACA"
inputs = tokenizer(seq, return_tensors="pt")
outputs = model(**inputs)

# if you want return attention weights
outputs = model(**inputs, output_attentions=True)
```

if you just want to get the last hidden states, please use:
```python
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    last_hidden_states = outputs.last_hidden_state
```
to make sure that the model is in evaluation mode without calculating gradients.

### Ultra fast embedding inference

#### Preare the data
Prepare a fasta file, same format as the `example/fasta/example_0.fasta` file. The fasta file should contain the sequences you want to embed. By running the following command, we will automatically collect all fasta files in the `example/fasta` directory and extract the embedding for each sequence.

### Run your inference
```bash
python unirna_tf/infer.py --fasta_path example/fasta --output_dir example/output --batch_size 1 --concurrency 1 --pretrained_path weights/unirna_L16
```
The `--concurrency` is the number of threads you want to use, corresponds to the number of GPUs you want to use. The `--batch_size` is the batch size for each thread, depending on the GPU RAM size of your machine. The `--pretrained_path` is the path to the pretrained model.

## Acknowledgments
Commercial inquiries, please contact [wenh@aisi.ac.cn](mailto:wenh@aisi.ac.cn)
