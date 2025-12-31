# Uni-RNA: The Large-Scale Pre-Trained Model for RNA Science

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a7fd7143501741b489ea2123a6ce3a50)](https://app.codacy.com/gh/ComDec/unirna_tf/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy - Coverage](https://app.codacy.com/project/badge/Coverage/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/ComDec/unirna_tf/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)

The Hugging Face compatible version of Uni-RNA, which is designed to be more efficient and easier to use.

## Installation

If you unzip the code from compressed file, please run `git init` to initialize the git repository. We need git info. to track the version of the code.

```bash
conda create -n unirna python=3.10
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

## Model summary

- Released checkpoints: L8 / L12 / L16. Default configs use 64-d attention heads, GELU feed-forward with width `3 * hidden_size`, rotary position embeddings, vocab size 10.

| Model | Layers | Hidden dim | Heads | FFN dim | Notes |
| --- | --- | --- | --- | --- | --- |
| L8  | 8  | 512  | 8  | 1536 | Fastest, lightest |
| L12 | 12 | 768  | 12 | 2304 | Balanced |
| L16 | 16 | 1024 | 16 | 3072 | Highest capacity |

- `UniRNAForMaskedLM` (a MaskedLM-style class): outputs the MLM head logits for each position (shape `[batch, seq_len+2, vocab_size]`), representing token prediction probabilities; `loss` is provided when labels are supplied.

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

# if you want return attention weights, set output_attentions=True
outputs = model(**inputs, output_attentions=True)

# if you want return the sequence embeddings without gradient:
with torch.no_grad():
    outputs = model(**inputs)
    sequence_embeddings = outputs.last_hidden_state
```


### Ultra fast embedding inference

#### Preare the data
Prepare a fasta file, same format as the `example/fasta/example_0.fasta` file. The fasta file should contain the sequences you want to embed. By running the following command, we will automatically collect all fasta files in the `example/fasta` directory and extract the embedding for each sequence.


#### Run your inference
```bash
python unirna_tf/infer.py --fasta_path example/fasta --output_dir example/output --batch_size 1 --concurrency 1 --pretrained_path weights/unirna_L16
```
The `--concurrency` is the number of threads you want to use, corresponds to the number of GPUs you want to use. The `--batch_size` is the batch size for each thread, depending on the GPU RAM size of your machine. The `--pretrained_path` is the path to the pretrained model.


## Acknowledgments
Commercial inquiries, please contact [wenh@aisi.ac.cn](mailto:wenh@aisi.ac.cn)
