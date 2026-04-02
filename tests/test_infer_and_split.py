import pytest
import torch
from Bio import SeqIO

from unirna_tf.infer import (
    fasta_output_stem,
    format_prediction_batch,
    prepare_seq,
    prepare_seq_dict,
    resolve_fasta_inputs,
)
from unirna_tf.split import split_fasta


def test_prepare_seq_functions(tmp_path):
    fasta_path = tmp_path / "sample.fasta"
    fasta_path.write_text(">seq1\nACGU\n>seq2\nGGAA\n")

    seqs = prepare_seq(str(fasta_path))
    assert seqs == ["ACGU", "GGAA"]

    seq_dicts = prepare_seq_dict(str(fasta_path))
    assert seq_dicts == [{"seq": "ACGU"}, {"seq": "GGAA"}]


def test_split_fasta_creates_files(tmp_path):
    fasta_path = tmp_path / "sample.fasta"
    fasta_path.write_text(">seq1\nACGU\n>seq2\nGGAA\n>seq3\nCCUU\n")

    out_dir = tmp_path / "splits"
    split_fasta(str(fasta_path), str(out_dir), 2)

    split_files = sorted(out_dir.glob("split_*.fasta"))
    assert len(split_files) == 2

    total_records = 0
    for file_path in split_files:
        records = list(SeqIO.parse(str(file_path), "fasta"))
        total_records += len(records)
        assert len(records) <= 2

    assert total_records == 3


def test_resolve_fasta_inputs_accepts_single_file(tmp_path):
    fasta_path = tmp_path / "sample.v1.fasta"
    fasta_path.write_text(">seq1\nACGU\n")

    assert resolve_fasta_inputs(str(fasta_path)) == [str(fasta_path)]


def test_resolve_fasta_inputs_filters_directory_entries(tmp_path):
    fasta_dir = tmp_path / "inputs"
    fasta_dir.mkdir()
    first = fasta_dir / "first.v1.fasta"
    second = fasta_dir / "second.fa.gz"
    ignored = fasta_dir / "notes.txt"
    nested = fasta_dir / "nested"

    first.write_text(">seq1\nACGU\n")
    second.write_text(">seq2\nGGAA\n")
    ignored.write_text("not fasta\n")
    nested.mkdir()
    (nested / "inside.fasta").write_text(">seq3\nCCUU\n")

    assert resolve_fasta_inputs(str(fasta_dir)) == [str(first), str(second)]


def test_resolve_fasta_inputs_rejects_colliding_output_names(tmp_path):
    fasta_dir = tmp_path / "inputs"
    fasta_dir.mkdir()
    (fasta_dir / "sample.fa").write_text(">seq1\nACGU\n")
    (fasta_dir / "sample.fasta").write_text(">seq2\nGGAA\n")

    with pytest.raises(ValueError, match="overwrite the same output"):
        resolve_fasta_inputs(str(fasta_dir))


def test_fasta_output_stem_preserves_earlier_dots():
    assert fasta_output_stem("/tmp/sample.v1.fasta") == "sample.v1"
    assert fasta_output_stem("/tmp/sample.v2.fa.gz") == "sample.v2"


def test_format_prediction_batch_whole_seq_includes_padding_metadata():
    class DummyPredictions:
        def __init__(self):
            self.pooler_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            self.last_hidden_state = torch.tensor(
                [
                    [[10.0, 11.0], [12.0, 13.0], [0.0, 0.0]],
                    [[20.0, 21.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )

    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    batch = format_prediction_batch(DummyPredictions(), attention_mask, save_whole_seq=True)

    assert set(batch) == {"pooler_output", "last_hidden_state", "attention_mask", "sequence_lengths"}
    assert torch.equal(batch["attention_mask"], attention_mask)
    assert torch.equal(batch["sequence_lengths"], torch.tensor([2, 1]))
    assert batch["pooler_output"].dtype == torch.float32
    assert batch["last_hidden_state"].dtype == torch.float32


def test_format_prediction_batch_pooled_mode_returns_tensor():
    class DummyPredictions:
        def __init__(self):
            self.pooler_output = torch.tensor([[1.0, 2.0]])

    attention_mask = torch.tensor([[1, 1]])
    batch = format_prediction_batch(DummyPredictions(), attention_mask, save_whole_seq=False)

    assert torch.equal(batch, torch.tensor([[1.0, 2.0]]))


@pytest.mark.parametrize("num_files", [0, -1])
def test_split_fasta_rejects_non_positive_shard_counts(tmp_path, num_files):
    fasta_path = tmp_path / "sample.fasta"
    fasta_path.write_text(">seq1\nACGU\n")

    with pytest.raises(ValueError, match="positive integer"):
        split_fasta(str(fasta_path), str(tmp_path / "splits"), num_files)


def test_split_fasta_caps_output_count_to_records(tmp_path):
    fasta_path = tmp_path / "sample.fasta"
    fasta_path.write_text(">seq1\nACGU\n>seq2\nGGAA\n")

    out_dir = tmp_path / "splits"
    split_fasta(str(fasta_path), str(out_dir), 5)

    split_files = sorted(out_dir.glob("split_*.fasta"))
    assert len(split_files) == 2
    assert [len(list(SeqIO.parse(str(file_path), "fasta"))) for file_path in split_files] == [1, 1]


def test_split_fasta_rejects_empty_input(tmp_path):
    fasta_path = tmp_path / "empty.fasta"
    fasta_path.write_text("")

    with pytest.raises(ValueError, match="has no records"):
        split_fasta(str(fasta_path), str(tmp_path / "splits"), 2)
