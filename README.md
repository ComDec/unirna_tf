---
title: README
authors:
    - Xi Wang
---

# README

The light version of UniRNA, which is designed to be more efficient and easier to use. The light version is trained with few codes, and is suitable for small-scale applications and experiments.

## 安装

```bash
pip install .
```

## 使用

We provide jupyter notebook to demonstrate how to use the pretrained model. You can find the notebook in the `examples` directory.

### Transformers

```python
import unirna
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("unirna_L16_E1024_DPRNA500M_STEP400K")
model = AutoModel.from_pretrained("unirna_L16_E1024_DPRNA500M_STEP400K")
```