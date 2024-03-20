---
title: README
authors:
    - Xi Wang
---

# README

The light version of Uni-RNA, which is designed to be more efficient and easier to use. The light version is trained with few codes, and is suitable for small-scale applications and experiments.

## Installation

```bash
pip install .
```

## How to use

We provide jupyter notebook to demonstrate how to use the pretrained model. You can find the notebook in the `examples` directory.

### Transformers

```python
import unirna
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("unirna_L16_E1024_DPRNA500M_STEP400K")
model = AutoModel.from_pretrained("unirna_L16_E1024_DPRNA500M_STEP400K")

seq = "AUCGGUGACA"
inputs = tokenizer(seq, return_tensors="pt")
outputs = model(**inputs)

# if you want return attention weights
outputs = model(**inputs, output_attentions=True)

```
