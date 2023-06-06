---
title: README
authors:
    - Zhiyuan Chen
date: 2023-06-06
---

# README

本项目将一个UniRNA Checkpoint转换成一个HuggingFace Transformers兼容的Pretrained。

## 安装

```bash
pip install .
```
 
## 转换

```
python -m unirna.convert unirna_L16_E1024_DPRNA500M_STEP400K.pt 
```

对于预训练的Checkpoint，在本例中使用`unirna_L16_E1024_DPRNA500M_STEP400K.pt`。
`convert`将会自动识别模型结构参数，生成恰当的配置文件，并转换模型结构。
最终结果将保存在同名（但没有扩展名）的目录中，本例为`unirna_L16_E1024_DPRNA500M_STEP400K`。

## 使用

### DeepProtein

在DeepProtein训练时，请在`--sequence.pretrained`指定转换后的文件路径，建议指定绝对路径。

```
python -m deepprotein.train --sequence.pretrained /path/to/unirna_L16_E1024_DPRNA500M_STEP400K
```

### Transformers

在通过transformers使用转换后的Pretrained时，请务必`import unirna`来确保配置、模型和令牌器被正确注册。

```python
import unirna  # import的顺序不重要
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("unirna_L16_E1024_DPRNA500M_STEP400K")
model = AutoModel.from_pretrained("unirna_L16_E1024_DPRNA500M_STEP400K")
```

## 文件结构

```bash
- {unirna}
-   |- convert.py
-   |- config.py
-   |- model.py
-   |- tokenizer.py
-   |- template
-     |- vocab.txt
-     |- tokenizer_config.json
-     |- special_tokens_map.json
```


