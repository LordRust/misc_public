"""Tests for the standalone cgMLST missing-data analysis utility."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "cgmlst_missing_data.py"
SPEC = importlib.util.spec_from_file_location("cgmlst_missing_data", SCRIPT)
assert SPEC and SPEC.loader
cg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cg
SPEC.loader.exec_module(cg)


def write_profiles(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def make_nontransitive(tmp_path: Path) -> cg.PairwiseCache:
    path = write_profiles(
        tmp_path / "profiles.tsv",
        "#Name\tST\tl1\tl2\tl3\tl4\n"
        "A\t8\t1\t-\t3\t5\n"
        "B\t8\t1\t2\t-\t5\n"
        "C\t8\t-\t2\t4\t5\n",
    )
    data = cg.read_profiles(path, ["ST"])
    d_values, c_values = cg.calculate_pairwise(data.profiles)
    return cg.PairwiseCache(**data.__dict__, distances=d_values, comparable=c_values)


def test_metadata_is_explicit_and_distances_are_correct(tmp_path: Path):
    path = write_profiles(
        tmp_path / "profiles.txt",
        "#ID ST locus1 locus2\nA 1 10 -\nB 2 10 7\nC 2 11 7\n",
    )
    excluded = cg.read_profiles(path, ["ST"])
    d_values, c_values = cg.calculate_pairwise(excluded.profiles)
    assert list(d_values) == [0, 1, 1]
    assert list(c_values) == [1, 1, 2]
    assert excluded.id_header == "#ID"
    assert excluded.locus_names.tolist() == ["locus1", "locus2"]

    included = cg.read_profiles(path, [])
    included_d, _ = cg.calculate_pairwise(included.profiles)
    assert list(included_d) == [1, 2, 1]
    assert included.locus_names.tolist()[0] == "ST"


def test_zero_distance_contradiction_and_complete_case_loss(tmp_path: Path):
    cache = make_nontransitive(tmp_path)
    # AB=(0,2), AC=(1,2), BC=(0,2)
    assert list(cache.distances) == [0, 1, 0]
    assert list(cache.comparable) == [2, 2, 2]
    result = cg.zero_component_results(cache, (0, 1, 2), required_c=1)
    assert len(result) == 1
    assert result[0]["contradictory_pairs"] == 1
    assert result[0]["unverified_pairs"] == 0

    complete = cg.complete_case_result(cache, (0, 1, 2))
    assert complete["complete_case_loci"] == 1
    assert complete["observed_polymorphic_loci"] == 1
    assert complete["polymorphic_loci_lost"] == 1
    assert complete["pairs_distance_decreased"] == 1
    assert complete["previously_distinguishable_pairs_now_identical"] == 1
    assert complete["complete_case_status"] == "validated"


def test_unverified_pair_in_zero_component(tmp_path: Path):
    path = write_profiles(
        tmp_path / "unverified.tsv",
        "#Name\tl1\tl2\nA\t1\t-\nB\t1\t2\nC\t-\t2\n",
    )
    data = cg.read_profiles(path, [])
    distances, comparable = cg.calculate_pairwise(data.profiles)
    cache = cg.PairwiseCache(
        **data.__dict__, distances=distances, comparable=comparable
    )
    assert list(comparable) == [1, 0, 1]
    detail = cg.zero_component_results(cache, (0, 1, 2), required_c=1)[0]
    assert detail["contradictory_pairs"] == 0
    assert detail["unverified_pairs"] == 1
    assert detail["unverified_samples"] == {0, 2}


def test_overlap_filters_change_connectivity(tmp_path: Path):
    cache = make_nontransitive(tmp_path)
    low = cg.cluster_at_thresholds(cache, [0], required_c=1)[0]
    high = cg.cluster_at_thresholds(cache, [0], required_c=3)[0]
    assert sorted(map(len, low)) == [3]
    assert sorted(map(len, high)) == [1, 1, 1]
    assert cg.required_comparable(4, None, None) == 1
    assert cg.required_comparable(4, None, 0.51) == 3


def test_zero_retained_loci_and_leave_one_out(tmp_path: Path):
    path = write_profiles(tmp_path / "none.tsv", "#Name\tl1\tl2\nA\t1\t-\nB\t-\t2\n")
    data = cg.read_profiles(path, [])
    distances, comparable = cg.calculate_pairwise(data.profiles)
    cache = cg.PairwiseCache(
        **data.__dict__, distances=distances, comparable=comparable
    )
    result = cg.complete_case_result(cache, (0, 1))
    assert result["complete_case_loci"] == 0
    assert result["complete_case_status"] == "not_evaluable"
    assert [row["loci_recovered_by_exclusion"] for row in result["leave_one_out"]] == [
        1,
        1,
    ]


def test_cache_round_trip_and_gzip_pair_export(tmp_path: Path):
    cache = make_nontransitive(tmp_path)
    cache_path = tmp_path / "cache.npz"
    cg.save_cache(cache_path, cache)
    loaded = cg.load_cache(cache_path)
    np.testing.assert_array_equal(loaded.profiles, cache.profiles)
    np.testing.assert_array_equal(loaded.distances, cache.distances)
    assert loaded.allele_levels == cache.allele_levels
    assert loaded.input_sha256 == cache.input_sha256

    pairs_path = tmp_path / "pairs.csv.gz"
    cg.write_pair_csv(pairs_path, loaded)
    with gzip.open(pairs_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[1]["sample_a"] == "A"
    assert rows[1]["sample_b"] == "C"
    assert rows[1]["D"] == "1"


def test_pseudonymized_profiles_are_consecutive_and_used_by_cache(tmp_path: Path):
    input_path = write_profiles(
        tmp_path / "duplicates.tsv",
        "#ID\tST\tl1\noriginal\t1\t10\noriginal\t2\t11\nthird\t3\t12\n",
    )
    cache_path = tmp_path / "profiles.npz"
    assert (
        cg.main(
            [
                "compute",
                str(input_path),
                "--metadata-column",
                "ST",
                "--cache",
                str(cache_path),
                "--pseudonym-prefix",
                "isolate",
            ]
        )
        == 0
    )
    pseudonymized = tmp_path / "profiles.pseudonymized.tsv"
    with pseudonymized.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows[0] == ["#ID", "ST", "l1"]
    assert [row[0] for row in rows[1:]] == [
        "isolate-0001",
        "isolate-0002",
        "isolate-0003",
    ]
    assert [row[1:] for row in rows[1:]] == [["1", "10"], ["2", "11"], ["3", "12"]]
    assert cg.load_cache(cache_path).names.tolist() == [
        "isolate-0001",
        "isolate-0002",
        "isolate-0003",
    ]


def test_full_cli_analysis_outputs_and_bounded_witnesses(tmp_path: Path):
    input_path = write_profiles(
        tmp_path / "profiles.tsv",
        "#Name\tST\tl1\tl2\tl3\tl4\n"
        "A\t8\t1\t-\t3\t5\n"
        "B\t8\t1\t2\t-\t5\n"
        "C\t8\t-\t2\t4\t5\n",
    )
    cache_path = tmp_path / "profiles.npz"
    assert (
        cg.main(
            [
                "compute",
                str(input_path),
                "--metadata-column",
                "ST",
                "--cache",
                str(cache_path),
            ]
        )
        == 0
    )
    output = tmp_path / "analysis"
    assert (
        cg.main(
            [
                "analyze",
                str(cache_path),
                "--thresholds",
                "0,1",
                "--output-dir",
                str(output),
                "--representative-count",
                "1",
                "--max-witnesses",
                "1",
            ]
        )
        == 0
    )
    expected = {
        "threshold_summary.csv",
        "clusters.csv",
        "cluster_members.csv",
        "cluster_size_summary.csv",
        "zero_components.csv",
        "zero_component_members.csv",
        "zero_issue_examples.csv",
        "complete_case_pair_changes.csv",
        "leave_one_out.csv",
        "representative_clusters.csv",
        "run_metrics.csv",
        "summary.txt",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with (output / "zero_issue_examples.csv").open(newline="") as handle:
        witnesses = list(csv.DictReader(handle))
    assert len(witnesses) == 1
    assert witnesses[0]["issue_type"] == "contradictory"
    with (output / "leave_one_out.csv").open(newline="") as handle:
        leave_rows = list(csv.DictReader(handle))
    assert leave_rows
    assert "loci_recovered_by_exclusion" in leave_rows[0]


def test_duplicate_ids_receive_unique_numeric_suffixes(tmp_path: Path):
    path = write_profiles(
        tmp_path / "duplicates.tsv",
        "#Name\tl1\nA\t1\nA\t2\nA\t3\nA-2\t4\nA-2\t5\n",
    )
    data = cg.read_profiles(path, [])
    assert data.names.tolist() == ["A", "A-2", "A-3", "A-2-2", "A-2-3"]


@pytest.mark.parametrize(
    "text,message",
    [("#Name\tl1\nA\t1\t2\n", "fields; expected")],
)
def test_invalid_input(tmp_path: Path, text: str, message: str):
    path = write_profiles(tmp_path / "invalid.tsv", text)
    with pytest.raises(cg.UserError, match=message):
        cg.read_profiles(path, [])
