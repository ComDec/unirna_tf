import argparse
import os
from typing import Dict

import numpy as np
import ray
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer

import unirna_tf
from unirna_tf import UniRNAModels


def prepare_seq(fasta_path):
    seq_list = []
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Prepare fasta file"):
        seq_list.append(str(record.seq))
    return np.asarray(seq_list)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        "-bsz",
        type=int,
        required=True,
        help="Batch size.",
    )

    parser.add_argument(
        "--concurrency",
        "-con",
        type=int,
        required=True,
        help="Number of concurrency to use for inference.",
    )

    parser.add_argument(
        "--fasta_path",
        "-fp",
        type=str,
        required=True,
        help="Path to the fasta file containing the sequences.",
    )

    parser.add_argument(
        "--pretrained_path",
        "-pp",
        type=str,
        required=True,
        help="Path to the pretrained model weights.",
    )

    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        required=True,
        help="Directory to save the inference results.",
    )

    parser.add_argument(
        "--temp_dir",
        "-td",
        type=str,
        default="/tmp/ray",
        help="Temporary directory to store the Ray logs.",
    )

    parser.add_argument(
        "--nums_array",
        "-na",
        type=int,
        help="Numbers of splits to divide the input fasta.",
    )

    return parser.parse_args()


class UniRNAPredictor:
    def __init__(self, pretrained_path: str):
        # Set "cuda:0" as the device so the Huggingface pipeline uses GPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model = UniRNAModels.from_pretrained(pretrained_path)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.model = self.model.to(torch.bfloat16)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        tokens = self.tokenizer(
            list(batch["data"]), padding=True, truncation=True, return_tensors="pt", max_length=4096
        )
        with torch.inference_mode():
            predictions = self.model(
                tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device), output_attentions=False
            )
        batch["output"] = predictions.pooler_output.float().cpu().numpy()
        return batch


def prepare_seq_partition(fasta_path, start, end):
    with open(fasta_path, "r") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if start <= i < end:
                yield str(record.seq)


def cli_main():
    import time

    start = time.time()
    args = parser_args()

    # region size calculation
    total_records = sum(1 for _ in SeqIO.parse(args.fasta_path, "fasta"))
    print(f"Total records: {total_records}")
    records_per_partition = total_records // args.nums_array

    # create output directorys
    for i in range(args.nums_array):
        os.makedirs(f"{args.output_dir}/part_{i}", exist_ok=True)

    ray.init(_temp_dir=args.temp_dir)

    for i in range(args.nums_array):
        start = i * records_per_partition
        end = start + records_per_partition if i < args.nums_array - 1 else total_records
        print(f"Processing part {i}, start: {start}, end: {end}")
        infer_array = list(prepare_seq_partition(args.fasta_path, start, end))
        ds = ray.data.from_numpy(np.asarray(infer_array))
        unirna_predictor = UniRNAPredictor(args.pretrained_path)

        prediction = ds.map_batches(
            unirna_predictor,
            num_gpus=1,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        )

        prediction.write_numpy(f"{args.output_dir}/part_{i}", column=["data", "output"])

    print("Time taken: ", time.time() - start)


if __name__ == "__main__":
    cli_main()
