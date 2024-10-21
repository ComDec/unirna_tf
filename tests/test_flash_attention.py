from typing import Dict

import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoTokenizer

import unirna_tf
from unirna_tf import UniRNAModels


def test_flash_attention():

    seq_list = []
    for record in SeqIO.parse("../example/rna_puzzle.fasta", "fasta"):
        seq_list.append(str(record.seq))

    tokenizer = AutoTokenizer.from_pretrained("../weights/unirna_L16_E1024_DPRNA500M_STEP400K")
    tokens = tokenizer(seq_list[:1], padding=True, return_tensors="pt").to("cuda")

    model = UniRNAModels.from_pretrained("../weights/unirna_L16_E1024_DPRNA500M_STEP400K")
    model = model.to("cuda")
    model = model.to(torch.bfloat16)

    import time

    start = time.time()
    with torch.no_grad():
        output_a = model(**tokens, output_attentions=False)
    print(time.time() - start)

    model = UniRNAModels.from_pretrained("../weights/unirna_L16_E1024_DPRNA500M_STEP400K_flash")
    model = model.to("cuda")
    model = model.to(torch.bfloat16)

    start = time.time()
    with torch.no_grad():
        output_b = model(**tokens, output_attentions=False)
    print(time.time() - start)

    assert (
        np.abs((output_a.pooler_output.float().cpu().numpy() - output_b.pooler_output.float().cpu().numpy())).mean()
        < 0.005
    )
