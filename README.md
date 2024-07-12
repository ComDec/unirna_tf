---
title: README
authors:
    - Xi Wang
---

# README

The light version of Uni-RNA, which is designed to be more efficient and easier to use. The light version is trained with few codes, and is suitable for small-scale applications and experiments.

## Installation

If you unzip the code from compressed file, please run `git init` to initialize the git repository. We need git info. to track the version of the code.

```bash
pip install .
```

## How to use

We provide jupyter notebook to demonstrate how to use the pretrained model. You can find the notebook in the `examples` directory. The model weights is stored in the `weights` directory.

### Transformers

```python
import unirna_tf
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./weights/unirna_L16_E1024_DPRNA500M_STEP400K")
model = AutoModel.from_pretrained("./weights/unirna_L16_E1024_DPRNA500M_STEP400K")

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
