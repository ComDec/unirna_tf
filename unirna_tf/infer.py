import argparse
import os
import tempfile
from typing import Dict, Optional

import ray
import torch
import torch.utils
from Bio import SeqIO
from datasets import Dataset
from ray.util.actor_pool import ActorPool
from tqdm import tqdm
from transformers import AutoTokenizer

from unirna_tf import UniRNAModels


def prepare_seq(fasta_path):
    """Read sequences from a FASTA file and return a list of strings."""
    seq_list = []
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Prepare fasta file"):
        seq_list.append(str(record.seq))
    return seq_list


def prepare_seq_dict(fasta_path):
    """Read sequences from FASTA and return list of dicts with key 'seq'."""
    seq_list = []
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Prepare fasta file"):
        seq_list.append({"seq": str(record.seq)})
    return seq_list


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-related CLI arguments."""
    parser.add_argument(
        "--pretrained_path",
        "-pp",
        type=str,
        required=True,
        help="Path to the pretrained model weights.",
    )
    parser.add_argument(
        "--batch_size",
        "-bsz",
        type=int,
        required=True,
        help="Batch size.",
    )
    parser.add_argument(
        "--max_seq_len",
        "-msl",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--whole_seq",
        "-ws",
        action="store_true",
        help="Whether to save the whole sequence embedding or not.",
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add data input/output CLI arguments."""
    parser.add_argument(
        "--fasta_path",
        "-fp",
        type=str,
        required=True,
        help="Path to the fasta file containing the sequences.",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        required=True,
        help="Directory to save the inference results.",
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Add runtime-related CLI arguments (concurrency, temp dir)."""
    parser.add_argument(
        "--concurrency",
        "-con",
        type=int,
        required=True,
        help="Number of concurrency to use for inference.",
    )
    parser.add_argument(
        "--temp_dir",
        "-td",
        type=str,
        default=None,
        help="Temporary directory to store the Ray logs.",
    )


def _resolve_temp_dir(temp_dir: Optional[str]) -> str:
    """Create or validate a temporary directory path for Ray runtime."""
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    return tempfile.mkdtemp(prefix="ray-")


def parser_args():
    """Parse CLI arguments for embedding inference."""
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_data_args(parser)
    add_runtime_args(parser)
    args = parser.parse_args()
    args.temp_dir = _resolve_temp_dir(args.temp_dir)
    return args


@ray.remote
class UniRNAPredictor:
    """Ray actor wrapper to run UniRNA inference on batches."""

    def __init__(
        self, model, pretrained_path: str, batch_size: int = 128, max_seq_len: int = 1024, save_whole_seq: bool = False
    ):
        # Set "cuda:0" as the device so the Huggingface pipeline uses GPU.
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.save_whole_seq = save_whole_seq
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model = model
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        if self.device == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            self.model = self.model.to(torch.bfloat16)

    def encode(self, examples):
        """Tokenize a batch of sequences."""
        tokens = self.tokenizer(
            examples["seq"], padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_len
        )
        return tokens

    def predict(self, fasta_path, output_dir) -> Dict[str, list]:
        """Run inference on a FASTA file and save embeddings."""
        seq_list_dict = prepare_seq_dict(fasta_path)
        dataset = Dataset.from_list(seq_list_dict)
        dataset.set_transform(self.encode)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        results = []
        print(f"Predicting {fasta_path}")
        fasta_basename = os.path.basename(fasta_path)
        fasta_name = fasta_basename.split(".")[0]

        for tokens in tqdm(dataloader, desc=f"Predicting {fasta_name}"):
            with torch.inference_mode():
                predictions = self.model(
                    tokens["input_ids"].to(self.device),
                    tokens["attention_mask"].to(self.device),
                    output_attentions=False,
                )
            if self.save_whole_seq:
                results.append((predictions.pooler_output.float().cpu(), predictions.last_hidden_state.float().cpu()))
            else:
                results.append(predictions.pooler_output.float().cpu())

        torch.save(results, f"{output_dir}/{fasta_name}.pt")

        return len(results)


def cli_main():
    """Entry point for multi-actor embedding inference."""
    import time

    start = time.time()
    args = parser_args()
    file_paths = []
    for file_name in os.listdir(args.fasta_path):
        file_paths.append(os.path.join(args.fasta_path, file_name))

    num_actors = args.concurrency

    os.makedirs(args.output_dir, exist_ok=True)

    ray.init(_temp_dir=args.temp_dir)
    model = UniRNAModels.from_pretrained(args.pretrained_path)
    model_ref = ray.put(model)
    num_gpus = 1 if torch.cuda.is_available() else 0
    actors = [
        UniRNAPredictor.options(num_gpus=num_gpus, num_cpus=8).remote(
            model_ref, args.pretrained_path, args.batch_size, args.max_seq_len, args.whole_seq
        )
        for _ in range(num_actors)
    ]
    pool = ActorPool(actors)

    for file in file_paths:
        pool.submit(lambda a, f: a.predict.remote(f, args.output_dir), file)

    while pool.has_next():
        print(f"Finished prediction with {pool.get_next()} batches.")

    print("Time taken: ", time.time() - start)


if __name__ == "__main__":
    cli_main()
