# README

The light version of Uni-RNA, which is designed to be more efficient and easier to use. The light version is trained with few codes, and is suitable for small-scale applications and experiments.

## Installation

If you unzip the code from compressed file, please run `git init` to initialize the git repository. We need git info. to track the version of the code.

```bash
conda create -n unirna python=3.10
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

## How to use

We provide jupyter notebook to demonstrate how to use the pretrained model. You can find the notebook in the `examples` directory. The model weights is stored in the `weights` directory.

### Quick Start

**!!! You must convert string to uppercase before inputting the sequence to the model !!!**

Sequence "ATcg" is different from "ATCG", all the lowercase letters will be merged and converted to `unk_token` in the tokenizer.

```python
import unirna_tf
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./weights/unirna_L16")
model = AutoModel.from_pretrained("./weights/unirna_L16)

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
Prepare a fasta file, same format as the `data/fasta/example.fasta` file. The fasta file should contain the sequences you want to embed. By running the following command, we will automatically collect all fasta files in the `data/fasta` directory and extract the embedding for each sequence.

### Run your inference
```bash
python infer.py --fasta_path data/fasta --output_dir data/example --batch_size 32 --concurrency 8 --pretrained_path weights/unirna_L16
```
The `--concurrency` is the number of threads you want to use, corresponds to the number of GPUs you want to use. The `--batch_size` is the batch size for each thread, depending on the GPU RAM size of your machine. The `--pretrained_path` is the path to the pretrained model.
