from Bio import SeqIO

from unirna_tf.infer import prepare_seq, prepare_seq_dict
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
