---
title: README
authors:
    - Zhiyuan Chen
date: 2023-06-06
---

# README

本项目将一个UniRNA Checkpoint转换成一个HuggingFace Transformers兼容的Pretrained。
 
## 用法

```
python convert_ckpt.py unirna_L16_E1024_DPRNA500M_STEP400K.pt 
```

对于预训练的Checkpoint，在本例中使用unirna_L16_E1024_DPRNA500M_STEP400K.pt。
`convert_ckpt.py`将会自动识别模型结构参数，生成恰当的配置文件，并转换模型结构。
最终结果将保存在`unirna_L16_E1024_DPRNA500M_STEP400K`文件夹中。

在DeepProtein训练时，请在sequence.pretrained指定该文件夹，建议指定绝对路径。

```
python -m deepprotein.train --sequence.pretrained /path/to/unirna_L16_E1024_DPRNA500M_STEP400K
```

## 文件结构

```bash
- {unirna_transformers}
-   |- convert_ckpt.py
-   |- config.py
-   |- model.py
-   |- tokenizer.py
-   |- template
-     |- vocab.txt
-     |- tokenizer_config.json
-     |- special_tokens_map.json
```


