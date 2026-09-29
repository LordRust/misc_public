#!/usr/bin/env python3
"""Investigate the effects of missing data in categorical cgMLST profiles.

The program deliberately has no dependencies on the Bonsai application.  It has
two stages so the expensive pairwise calculation can be reused::

    python scripts/cgmlst_missing_data.py compute profiles.tsv \
        --metadata-column ST --cache mrsa.npz
    python scripts/cgmlst_missing_data.py analyze mrsa.npz \
        --thresholds 0,1,2,5 --output-dir mrsa-analysis

All columns after the first identifier column are treated as loci unless they
are explicitly named with ``--metadata-column``.  Alleles are categorical and
``-`` means missing. Repeated identifiers are made unique in input order by
appending ``-2``, ``-3``, and so on.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import resource
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


CACHE_SCHEMA = 1
MISSING = "-"


class UserError(ValueError):
    """An error in user input suitable for display without a traceback."""


@dataclass
class ProfileData:
    """Encoded input profiles and the information needed to decode them."""

    id_header: str
    names: np.ndarray
    metadata_names: np.ndarray
    metadata_values: np.ndarray
    locus_names: np.ndarray
    profiles: np.ndarray
    allele_levels: list[list[str]]
    input_sha256: str
    source: str


@dataclass
class PairwiseCache(ProfileData):
    """Profile data plus condensed pairwise distance and overlap arrays."""

    distances: np.ndarray
    comparable: np.ndarray


class UnionFind:
    """Small union-find implementation used for connected components."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on Linux/macOS."""

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_line(line: str, tab_delimited: bool) -> list[str]:
    if tab_delimited:
        return [field.strip() for field in line.rstrip("\r\n").split("\t")]
    return line.split()


def _nonempty_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, line


def read_profiles(path: Path, metadata_columns: Sequence[str]) -> ProfileData:
    """Read and encode a tab- or whitespace-delimited profile table.

    The file is scanned twice to avoid retaining millions of Python strings.
    """

    if not path.is_file():
        raise UserError(f"input file does not exist: {path}")

    lines = _nonempty_lines(path)
    try:
        header_line_number, header_line = next(lines)
    except StopIteration as error:
        raise UserError("input file is empty") from error
    tab_delimited = "\t" in header_line
    header = _split_line(header_line, tab_delimited)
    if len(header) < 2:
        raise UserError("input must contain an identifier and at least one locus")
    if any(not value for value in header):
        raise UserError("empty column names are not supported; use '-' only for data")
    if len(set(header)) != len(header):
        raise UserError("input contains duplicate column names")

    requested_metadata = list(metadata_columns)
    if len(set(requested_metadata)) != len(requested_metadata):
        raise UserError("the same --metadata-column was supplied more than once")
    unknown_metadata = sorted(set(requested_metadata) - set(header[1:]))
    if unknown_metadata:
        raise UserError("unknown metadata column(s): " + ", ".join(unknown_metadata))
    if header[0] in requested_metadata:
        raise UserError("the identifier column cannot also be metadata")

    metadata_indexes = [header.index(name) for name in requested_metadata]
    metadata_index_set = set(metadata_indexes)
    locus_indexes = [
        index for index in range(1, len(header)) if index not in metadata_index_set
    ]
    if not locus_indexes:
        raise UserError("no locus columns remain after excluding metadata")

    sample_count = 0
    for line_number, line in lines:
        fields = _split_line(line, tab_delimited)
        if len(fields) != len(header):
            raise UserError(
                f"line {line_number} has {len(fields)} fields; expected {len(header)}"
            )
        name = fields[0]
        if not name:
            raise UserError(f"line {line_number} has an empty sample identifier")
        sample_count += 1
    if sample_count == 0:
        raise UserError("input contains a header but no samples")

    profiles = np.zeros((sample_count, len(locus_indexes)), dtype=np.uint32)
    names: list[str] = []
    metadata_rows: list[list[str]] = []
    encoders: list[dict[str, int]] = [dict() for _ in locus_indexes]
    allele_levels: list[list[str]] = [[] for _ in locus_indexes]
    used_names: set[str] = set()
    next_suffix: dict[str, int] = {}

    lines = _nonempty_lines(path)
    next(lines)  # header, already validated
    for row_index, (_line_number, line) in enumerate(lines):
        fields = _split_line(line, tab_delimited)
        original_name = fields[0]
        unique_name = original_name
        if unique_name in used_names:
            suffix = next_suffix.get(original_name, 2)
            unique_name = f"{original_name}-{suffix}"
            while unique_name in used_names:
                suffix += 1
                unique_name = f"{original_name}-{suffix}"
            next_suffix[original_name] = suffix + 1
        else:
            next_suffix.setdefault(original_name, 2)
        used_names.add(unique_name)
        names.append(unique_name)
        metadata_rows.append([fields[index] for index in metadata_indexes])
        for output_index, input_index in enumerate(locus_indexes):
            allele = fields[input_index]
            if allele == MISSING:
                continue
            encoder = encoders[output_index]
            code = encoder.get(allele)
            if code is None:
                code = len(encoder) + 1
                encoder[allele] = code
                allele_levels[output_index].append(allele)
            profiles[row_index, output_index] = code

    metadata_values = np.asarray(metadata_rows, dtype=np.str_)
    if not requested_metadata:
        metadata_values = np.empty((sample_count, 0), dtype=np.str_)
    return ProfileData(
        id_header=header[0],
        names=np.asarray(names, dtype=np.str_),
        metadata_names=np.asarray(requested_metadata, dtype=np.str_),
        metadata_values=metadata_values,
        locus_names=np.asarray(
            [header[index] for index in locus_indexes], dtype=np.str_
        ),
        profiles=profiles,
        allele_levels=allele_levels,
        input_sha256=sha256_file(path),
        source=str(path.resolve()),
    )


def condensed_size(sample_count: int) -> int:
    return sample_count * (sample_count - 1) // 2


def row_offset(row: int, sample_count: int) -> int:
    """Index of pair (row, row + 1) in row-major condensed storage."""

    return row * sample_count - row * (row + 1) // 2


def pair_index(left: int, right: int, sample_count: int) -> int:
    if left == right:
        raise ValueError("a condensed matrix has no diagonal")
    if left > right:
        left, right = right, left
    return row_offset(left, sample_count) + right - left - 1


def calculate_pairwise(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate D and C using bounded 2-D vectorized work arrays."""

    sample_count = profiles.shape[0]
    length = condensed_size(sample_count)
    distances = np.empty(length, dtype=np.uint32)
    comparable = np.empty(length, dtype=np.uint32)
    for left in range(sample_count - 1):
        others = profiles[left + 1 :]
        called = (others != 0) & (profiles[left] != 0)
        start = row_offset(left, sample_count)
        stop = start + sample_count - left - 1
        comparable[start:stop] = np.count_nonzero(called, axis=1)
        distances[start:stop] = np.count_nonzero(
            called & (others != profiles[left]), axis=1
        )
    return distances, comparable


def pseudonymize_profiles(data: ProfileData, prefix: str = "sample") -> ProfileData:
    """Return profile data with stable consecutive row-order identifiers."""

    width = max(4, len(str(len(data.names))))
    names = np.asarray(
        [f"{prefix}-{index:0{width}d}" for index in range(1, len(data.names) + 1)],
        dtype=np.str_,
    )
    return replace(data, names=names)


def _atomic_target(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, delete=False
    )
    temp_path = Path(temp.name)
    temp.close()
    return temp_path


def _json_bytes(value: object) -> np.ndarray:
    return np.frombuffer(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )


def save_cache(path: Path, cache: PairwiseCache) -> None:
    temp_path = _atomic_target(path)
    try:
        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(CACHE_SCHEMA, dtype=np.uint32),
                id_header=np.asarray(cache.id_header, dtype=np.str_),
                names=cache.names,
                metadata_names=cache.metadata_names,
                metadata_values=cache.metadata_values,
                locus_names=cache.locus_names,
                profiles=cache.profiles,
                allele_levels_json=_json_bytes(cache.allele_levels),
                input_sha256=np.asarray(cache.input_sha256, dtype=np.str_),
                source=np.asarray(cache.source, dtype=np.str_),
                distances=cache.distances,
                comparable=cache.comparable,
            )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_cache(path: Path) -> PairwiseCache:
    if not path.is_file():
        raise UserError(f"cache does not exist: {path}")
    required = {
        "schema",
        "id_header",
        "names",
        "metadata_names",
        "metadata_values",
        "locus_names",
        "profiles",
        "allele_levels_json",
        "input_sha256",
        "source",
        "distances",
        "comparable",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise UserError(
                    "cache is missing fields: " + ", ".join(sorted(missing))
                )
            schema = int(data["schema"])
            if schema != CACHE_SCHEMA:
                raise UserError(
                    f"unsupported cache schema {schema}; expected {CACHE_SCHEMA}"
                )
            allele_levels = json.loads(data["allele_levels_json"].tobytes())
            cache = PairwiseCache(
                id_header=str(data["id_header"]),
                names=data["names"].copy(),
                metadata_names=data["metadata_names"].copy(),
                metadata_values=data["metadata_values"].copy(),
                locus_names=data["locus_names"].copy(),
                profiles=data["profiles"].copy(),
                allele_levels=allele_levels,
                input_sha256=str(data["input_sha256"]),
                source=str(data["source"]),
                distances=data["distances"].copy(),
                comparable=data["comparable"].copy(),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        if isinstance(error, UserError):
            raise
        raise UserError(f"could not read cache {path}: {error}") from error

    sample_count, locus_count = cache.profiles.shape
    expected_pairs = condensed_size(sample_count)
    if len(cache.names) != sample_count:
        raise UserError("cache has inconsistent sample names and profile dimensions")
    if len(cache.locus_names) != locus_count or len(cache.allele_levels) != locus_count:
        raise UserError("cache has inconsistent locus metadata")
    if cache.metadata_values.shape != (sample_count, len(cache.metadata_names)):
        raise UserError("cache has inconsistent metadata dimensions")
    if cache.distances.shape != (expected_pairs,) or cache.comparable.shape != (
        expected_pairs,
    ):
        raise UserError("cache has inconsistent condensed pair arrays")
    return cache


def write_pair_csv(path: Path, cache: PairwiseCache) -> None:
    temp_path = _atomic_target(path)
    # Preserve gzip behavior despite the random temporary suffix.
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(temp_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sample_a",
                    "sample_b",
                    "D",
                    "C",
                    "comparable_proportion",
                    "status",
                ]
            )
            locus_count = cache.profiles.shape[1]
            sample_count = len(cache.names)
            for left in range(sample_count - 1):
                start = row_offset(left, sample_count)
                for right in range(left + 1, sample_count):
                    index = start + right - left - 1
                    c_value = int(cache.comparable[index])
                    writer.writerow(
                        [
                            cache.names[left],
                            cache.names[right],
                            int(cache.distances[index]),
                            c_value,
                            format(c_value / locus_count, ".12g"),
                            "ok" if c_value else "no_joint_calls",
                        ]
                    )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    temp_path = _atomic_target(path)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_metrics(path: Path, rows: list[dict]) -> None:
    write_csv(path, ["stage", "elapsed_seconds", "peak_rss_bytes"], rows)


def metric(stage: str, started: float) -> dict:
    return {
        "stage": stage,
        "elapsed_seconds": f"{time.perf_counter() - started:.6f}",
        "peak_rss_bytes": peak_rss_bytes(),
    }


def required_comparable(
    locus_count: int, minimum_loci: int | None, minimum_proportion: float | None
) -> int:
    if minimum_loci is not None:
        return max(1, minimum_loci)
    if minimum_proportion is not None:
        return max(1, math.ceil(minimum_proportion * locus_count - 1e-12))
    return 1


def pair_eligible(c_value: int, required_c: int) -> bool:
    return c_value > 0 and c_value >= required_c


def components(
    union_find: UnionFind, members: Sequence[int] | None = None
) -> list[tuple[int, ...]]:
    if members is None:
        members = range(len(union_find.parent))
    grouped: dict[int, list[int]] = defaultdict(list)
    for member in members:
        grouped[union_find.find(member)].append(member)
    return [tuple(group) for group in grouped.values()]


def cluster_at_thresholds(
    cache: PairwiseCache, thresholds: Sequence[int], required_c: int
) -> dict[int, list[tuple[int, ...]]]:
    """Incrementally construct connected components at sorted thresholds."""

    sample_count = len(cache.names)
    union_find = UnionFind(sample_count)
    result: dict[int, list[tuple[int, ...]]] = {}
    previous = -1
    for threshold in thresholds:
        for left in range(sample_count - 1):
            start = row_offset(left, sample_count)
            stop = start + sample_count - left - 1
            row_d = cache.distances[start:stop]
            row_c = cache.comparable[start:stop]
            offsets = np.flatnonzero(
                (row_c >= required_c) & (row_d > previous) & (row_d <= threshold)
            )
            for offset in offsets:
                union_find.union(left, left + 1 + int(offset))
        groups = components(union_find)
        groups.sort(key=lambda group: tuple(sorted(str(cache.names[i]) for i in group)))
        result[threshold] = groups
        previous = threshold
    return result


def _pair_values(cache: PairwiseCache, left: int, right: int) -> tuple[int, int]:
    index = pair_index(left, right, len(cache.names))
    return int(cache.distances[index]), int(cache.comparable[index])


def _observed_polymorphic(profiles: np.ndarray) -> np.ndarray:
    result = np.zeros(profiles.shape[1], dtype=bool)
    for locus in range(profiles.shape[1]):
        values = profiles[:, locus]
        called = values[values != 0]
        if called.size > 1 and np.unique(called).size > 1:
            result[locus] = True
    return result


def complete_case_result(cache: PairwiseCache, members: tuple[int, ...]) -> dict:
    locus_count = cache.profiles.shape[1]
    subset = cache.profiles[np.asarray(members)]
    called = subset != 0
    missing_counts = np.count_nonzero(~called, axis=0)
    retained = missing_counts == 0
    retained_count = int(np.count_nonzero(retained))
    polymorphic = _observed_polymorphic(subset)
    retained_polymorphic = polymorphic & retained

    pair_cs: list[int] = []
    changes: list[dict] = []
    decreased = 0
    newly_identical = 0
    for position, left in enumerate(members[:-1]):
        for right in members[position + 1 :]:
            old_d, old_c = _pair_values(cache, left, right)
            pair_cs.append(old_c)
            new_d = int(
                np.count_nonzero(
                    cache.profiles[left, retained] != cache.profiles[right, retained]
                )
            )
            if new_d < old_d:
                decreased += 1
                became_identical = old_d > 0 and new_d == 0
                newly_identical += int(became_identical)
                changes.append(
                    {
                        "sample_a": str(cache.names[left]),
                        "sample_b": str(cache.names[right]),
                        "original_D": old_d,
                        "original_C": old_c,
                        "complete_case_D": new_d,
                        "complete_case_C": retained_count,
                        "became_identical": became_identical,
                    }
                )

    called_counts = np.count_nonzero(called, axis=1)
    leave_one_out: list[dict] = []
    if len(members) > 1:
        singleton_missing = missing_counts == 1
        for row, member in enumerate(members):
            recovered = int(np.count_nonzero(singleton_missing & ~called[row]))
            leave_one_out.append(
                {
                    "sample": str(cache.names[member]),
                    "called_loci": int(called_counts[row]),
                    "profile_completeness": format(
                        called_counts[row] / locus_count, ".12g"
                    ),
                    "baseline_complete_case_loci": retained_count,
                    "loci_recovered_by_exclusion": recovered,
                    "complete_case_loci_after_exclusion": retained_count + recovered,
                }
            )

    return {
        "minimum_pairwise_C": min(pair_cs) if pair_cs else "",
        "median_pairwise_C": statistics.median(pair_cs) if pair_cs else "",
        "minimum_profile_completeness": format(
            int(called_counts.min()) / locus_count, ".12g"
        ),
        "complete_case_loci": retained_count,
        "complete_case_percentage": format(retained_count / locus_count * 100, ".12g"),
        "observed_polymorphic_loci": int(np.count_nonzero(polymorphic)),
        "polymorphic_loci_retained": int(np.count_nonzero(retained_polymorphic)),
        "polymorphic_loci_lost": int(np.count_nonzero(polymorphic & ~retained)),
        "pairs_distance_decreased": decreased,
        "previously_distinguishable_pairs_now_identical": newly_identical,
        "complete_case_status": "validated" if retained_count else "not_evaluable",
        "changes": changes,
        "leave_one_out": leave_one_out,
        "retained_mask": retained,
    }


def zero_component_results(
    cache: PairwiseCache,
    members: tuple[int, ...],
    required_c: int,
    candidate_limit: int = 3,
) -> list[dict]:
    local = UnionFind(len(members))
    for left_pos, left in enumerate(members[:-1]):
        for right_pos in range(left_pos + 1, len(members)):
            right = members[right_pos]
            d_value, c_value = _pair_values(cache, left, right)
            if d_value == 0 and pair_eligible(c_value, required_c):
                local.union(left_pos, right_pos)

    output: list[dict] = []
    for group in components(local):
        if len(group) < 2:
            continue
        global_members = tuple(members[position] for position in group)
        contradictory = 0
        unverified = 0
        consistent = 0
        contradictory_samples: set[int] = set()
        unverified_samples: set[int] = set()
        candidates: dict[str, list[tuple[int, int]]] = {
            "contradictory": [],
            "unverified": [],
        }
        direct_cs: list[int] = []
        for left_pos, left in enumerate(global_members[:-1]):
            for right in global_members[left_pos + 1 :]:
                d_value, c_value = _pair_values(cache, left, right)
                direct_cs.append(c_value)
                if d_value > 0:
                    contradictory += 1
                    contradictory_samples.update((left, right))
                    if len(candidates["contradictory"]) < candidate_limit:
                        candidates["contradictory"].append((left, right))
                elif not pair_eligible(c_value, required_c):
                    unverified += 1
                    unverified_samples.update((left, right))
                    if len(candidates["unverified"]) < candidate_limit:
                        candidates["unverified"].append((left, right))
                else:
                    consistent += 1
        output.append(
            {
                "members": global_members,
                "contradictory_pairs": contradictory,
                "unverified_pairs": unverified,
                "verified_consistent_pairs": consistent,
                "minimum_direct_C": min(direct_cs),
                "median_direct_C": statistics.median(direct_cs),
                "contradictory_samples": contradictory_samples,
                "unverified_samples": unverified_samples,
                "candidates": candidates,
            }
        )
    output.sort(key=lambda item: tuple(str(cache.names[i]) for i in item["members"]))
    return output


def shortest_zero_path(
    cache: PairwiseCache,
    component_members: Sequence[int],
    start: int,
    target: int,
    required_c: int,
) -> list[int]:
    adjacency: dict[int, list[int]] = {member: [] for member in component_members}
    for left_pos, left in enumerate(component_members[:-1]):
        for right in component_members[left_pos + 1 :]:
            d_value, c_value = _pair_values(cache, left, right)
            if d_value == 0 and pair_eligible(c_value, required_c):
                adjacency[left].append(right)
                adjacency[right].append(left)
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda index: str(cache.names[index]))
    queue = deque([start])
    previous: dict[int, int | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == target:
            break
        for neighbor in adjacency[current]:
            if neighbor not in previous:
                previous[neighbor] = current
                queue.append(neighbor)
    if target not in previous:
        raise RuntimeError("zero-component endpoints unexpectedly have no zero path")
    path = []
    current: int | None = target
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return path


def _format_path(cache: PairwiseCache, path: Sequence[int]) -> tuple[str, str]:
    names = " -> ".join(str(cache.names[index]) for index in path)
    edges = []
    for left, right in zip(path, path[1:]):
        d_value, c_value = _pair_values(cache, left, right)
        edges.append(
            f"{cache.names[left]}--{cache.names[right]}:D={d_value},C={c_value}"
        )
    return names, ";".join(edges)


def write_profile_subset(
    path: Path, cache: ProfileData, members: Sequence[int]
) -> None:
    temp_path = _atomic_target(path)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [cache.id_header]
                + cache.metadata_names.tolist()
                + cache.locus_names.tolist()
            )
            for member in members:
                alleles = []
                for locus, code in enumerate(cache.profiles[member]):
                    alleles.append(
                        MISSING
                        if code == 0
                        else cache.allele_levels[locus][int(code) - 1]
                    )
                writer.writerow(
                    [cache.names[member]]
                    + cache.metadata_values[member].tolist()
                    + alleles
                )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_subset_pairs(
    path: Path, cache: PairwiseCache, members: Sequence[int]
) -> None:
    rows = []
    locus_count = cache.profiles.shape[1]
    for left_pos, left in enumerate(members[:-1]):
        for right in members[left_pos + 1 :]:
            d_value, c_value = _pair_values(cache, left, right)
            rows.append(
                {
                    "sample_a": str(cache.names[left]),
                    "sample_b": str(cache.names[right]),
                    "D": d_value,
                    "C": c_value,
                    "comparable_proportion": format(c_value / locus_count, ".12g"),
                    "status": "ok" if c_value else "no_joint_calls",
                }
            )
    write_csv(
        path,
        ["sample_a", "sample_b", "D", "C", "comparable_proportion", "status"],
        rows,
    )


def analyze_cache(
    cache: PairwiseCache,
    thresholds: Sequence[int],
    output_dir: Path,
    minimum_loci: int | None,
    minimum_proportion: float | None,
    representative_count: int,
    max_witnesses: int,
) -> None:
    locus_count = cache.profiles.shape[1]
    required_c = required_comparable(locus_count, minimum_loci, minimum_proportion)
    cluster_sets = cluster_at_thresholds(cache, thresholds, required_c)

    cluster_rows: list[dict] = []
    member_rows: list[dict] = []
    zero_rows: list[dict] = []
    zero_member_rows: list[dict] = []
    change_rows: list[dict] = []
    leave_rows: list[dict] = []
    cluster_records: list[dict] = []
    complete_cache: dict[tuple[int, ...], dict] = {}

    for threshold in thresholds:
        for ordinal, raw_members in enumerate(cluster_sets[threshold], 1):
            members = tuple(
                sorted(raw_members, key=lambda index: str(cache.names[index]))
            )
            cluster_id = f"t{threshold}_c{ordinal:04d}"
            membership_key = tuple(sorted(members))
            complete = complete_cache.get(membership_key)
            if complete is None:
                complete = complete_case_result(cache, membership_key)
                complete_cache[membership_key] = complete
            zero_components = zero_component_results(
                cache, membership_key, required_c, candidate_limit=max_witnesses
            )
            contradiction_count = sum(
                item["contradictory_pairs"] for item in zero_components
            )
            unverified_count = sum(item["unverified_pairs"] for item in zero_components)
            problematic_samples = (
                set().union(
                    *(
                        item["contradictory_samples"] | item["unverified_samples"]
                        for item in zero_components
                    )
                )
                if zero_components
                else set()
            )

            cluster_row = {
                "threshold": threshold,
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "minimum_pairwise_C": complete["minimum_pairwise_C"],
                "median_pairwise_C": complete["median_pairwise_C"],
                "minimum_profile_completeness": complete[
                    "minimum_profile_completeness"
                ],
                "complete_case_loci": complete["complete_case_loci"],
                "complete_case_percentage": complete["complete_case_percentage"],
                "observed_polymorphic_loci": complete["observed_polymorphic_loci"],
                "polymorphic_loci_retained": complete["polymorphic_loci_retained"],
                "polymorphic_loci_lost": complete["polymorphic_loci_lost"],
                "pairs_distance_decreased": complete["pairs_distance_decreased"],
                "previously_distinguishable_pairs_now_identical": complete[
                    "previously_distinguishable_pairs_now_identical"
                ],
                "complete_case_status": complete["complete_case_status"],
                "zero_components": len(zero_components),
                "contradictory_pairs": contradiction_count,
                "unverified_pairs": unverified_count,
                "affected_samples": len(problematic_samples),
            }
            cluster_rows.append(cluster_row)
            cluster_records.append(
                {
                    **cluster_row,
                    "members": membership_key,
                    "zero_details": zero_components,
                }
            )
            for member in members:
                member_rows.append(
                    {
                        "threshold": threshold,
                        "cluster_id": cluster_id,
                        "sample": str(cache.names[member]),
                    }
                )
            for component_ordinal, item in enumerate(zero_components, 1):
                component_id = f"{cluster_id}_z{component_ordinal:04d}"
                zero_rows.append(
                    {
                        "threshold": threshold,
                        "cluster_id": cluster_id,
                        "zero_component_id": component_id,
                        "component_size": len(item["members"]),
                        "contradictory_pairs": item["contradictory_pairs"],
                        "unverified_pairs": item["unverified_pairs"],
                        "verified_consistent_pairs": item["verified_consistent_pairs"],
                        "minimum_direct_C": item["minimum_direct_C"],
                        "median_direct_C": item["median_direct_C"],
                    }
                )
                for member in item["members"]:
                    issue_types = []
                    if member in item["contradictory_samples"]:
                        issue_types.append("contradictory")
                    if member in item["unverified_samples"]:
                        issue_types.append("unverified")
                    zero_member_rows.append(
                        {
                            "threshold": threshold,
                            "cluster_id": cluster_id,
                            "zero_component_id": component_id,
                            "sample": str(cache.names[member]),
                            "issue_types": ";".join(issue_types),
                        }
                    )
            for change in complete["changes"]:
                change_rows.append(
                    {"threshold": threshold, "cluster_id": cluster_id, **change}
                )
            for item in complete["leave_one_out"]:
                leave_rows.append(
                    {"threshold": threshold, "cluster_id": cluster_id, **item}
                )

    threshold_rows = []
    size_groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for threshold in thresholds:
        relevant = [row for row in cluster_rows if row["threshold"] == threshold]
        relevant_zero = [row for row in zero_rows if row["threshold"] == threshold]
        eligible_pairs = int(np.count_nonzero(cache.comparable >= required_c))
        problematic_clusters = sum(
            row["contradictory_pairs"] > 0 or row["unverified_pairs"] > 0
            for row in relevant
        )
        problematic_zero_components = sum(
            row["contradictory_pairs"] > 0 or row["unverified_pairs"] > 0
            for row in relevant_zero
        )
        zero_endpoint_pairs = sum(
            row["component_size"] * (row["component_size"] - 1) // 2
            for row in relevant_zero
        )
        contradictory_pairs = sum(row["contradictory_pairs"] for row in relevant)
        unverified_pairs = sum(row["unverified_pairs"] for row in relevant)
        threshold_rows.append(
            {
                "threshold": threshold,
                "required_comparable_loci": required_c,
                "eligible_pairs": eligible_pairs,
                "clusters": len(relevant),
                "non_singleton_clusters": sum(
                    row["cluster_size"] > 1 for row in relevant
                ),
                "largest_cluster": max(row["cluster_size"] for row in relevant),
                "problematic_clusters": problematic_clusters,
                "problematic_cluster_percentage": format(
                    problematic_clusters / len(relevant) * 100, ".12g"
                ),
                "zero_connected_components": len(relevant_zero),
                "problematic_zero_components": problematic_zero_components,
                "zero_component_endpoint_pairs": zero_endpoint_pairs,
                "contradictory_pairs": contradictory_pairs,
                "contradictory_pair_percentage": format(
                    (
                        contradictory_pairs / zero_endpoint_pairs * 100
                        if zero_endpoint_pairs
                        else 0
                    ),
                    ".12g",
                ),
                "unverified_pairs": unverified_pairs,
                "unverified_pair_percentage": format(
                    (
                        unverified_pairs / zero_endpoint_pairs * 100
                        if zero_endpoint_pairs
                        else 0
                    ),
                    ".12g",
                ),
                "polymorphic_loci_lost_sum": sum(
                    row["polymorphic_loci_lost"] for row in relevant
                ),
            }
        )
        for row in relevant:
            size_groups[(threshold, row["cluster_size"])].append(row)

    size_rows = []
    for (threshold, size), rows in sorted(size_groups.items()):
        problematic_count = sum(
            row["contradictory_pairs"] > 0 or row["unverified_pairs"] > 0
            for row in rows
        )
        size_rows.append(
            {
                "threshold": threshold,
                "cluster_size": size,
                "cluster_count": len(rows),
                "problematic_cluster_count": problematic_count,
                "problematic_cluster_percentage": format(
                    problematic_count / len(rows) * 100, ".12g"
                ),
                "contradictory_pairs": sum(row["contradictory_pairs"] for row in rows),
                "unverified_pairs": sum(row["unverified_pairs"] for row in rows),
                "distance_decreased_pairs": sum(
                    row["pairs_distance_decreased"] for row in rows
                ),
                "newly_identical_pairs": sum(
                    row["previously_distinguishable_pairs_now_identical"]
                    for row in rows
                ),
            }
        )

    # Select unique memberships so nested thresholds do not fill the representative set
    # with identical profile subsets.
    ranked = sorted(
        (
            record
            for record in cluster_records
            if record["contradictory_pairs"] or record["unverified_pairs"]
        ),
        key=lambda row: (
            -row["contradictory_pairs"],
            -row["unverified_pairs"],
            -row["cluster_size"],
            row["threshold"],
            row["cluster_id"],
        ),
    )
    representatives = []
    seen_memberships: set[tuple[int, ...]] = set()
    for record in ranked:
        if len(representatives) >= representative_count:
            break
        if record["members"] in seen_memberships:
            continue
        seen_memberships.add(record["members"])
        representatives.append(record)

    representative_rows = []
    witness_rows = []
    representative_dir = output_dir / "representative_clusters"
    representative_dir.mkdir(parents=True, exist_ok=True)
    for rank, record in enumerate(representatives, 1):
        prefix = f"representative_{rank:03d}"
        profile_file = representative_dir / f"{prefix}.profiles.tsv"
        pairs_file = representative_dir / f"{prefix}.pairs.csv"
        write_profile_subset(profile_file, cache, record["members"])
        write_subset_pairs(pairs_file, cache, record["members"])
        representative_rows.append(
            {
                "rank": rank,
                "threshold": record["threshold"],
                "cluster_id": record["cluster_id"],
                "cluster_size": record["cluster_size"],
                "contradictory_pairs": record["contradictory_pairs"],
                "unverified_pairs": record["unverified_pairs"],
                "profiles_file": str(profile_file.relative_to(output_dir)),
                "pairs_file": str(pairs_file.relative_to(output_dir)),
            }
        )
        emitted = defaultdict(int)
        for component_ordinal, detail in enumerate(record["zero_details"], 1):
            for issue_type in ("contradictory", "unverified"):
                candidates = sorted(
                    detail["candidates"][issue_type],
                    key=lambda pair: (
                        str(cache.names[pair[0]]),
                        str(cache.names[pair[1]]),
                    ),
                )
                for left, right in candidates:
                    if emitted[issue_type] >= max_witnesses:
                        break
                    path = shortest_zero_path(
                        cache, detail["members"], left, right, required_c
                    )
                    path_names, path_edges = _format_path(cache, path)
                    direct_d, direct_c = _pair_values(cache, left, right)
                    witness_rows.append(
                        {
                            "representative_rank": rank,
                            "threshold": record["threshold"],
                            "cluster_id": record["cluster_id"],
                            "zero_component_ordinal": component_ordinal,
                            "issue_type": issue_type,
                            "sample_a": str(cache.names[left]),
                            "sample_b": str(cache.names[right]),
                            "direct_D": direct_d,
                            "direct_C": direct_c,
                            "witness_path": path_names,
                            "witness_edges": path_edges,
                        }
                    )
                    emitted[issue_type] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "threshold_summary.csv", list(threshold_rows[0]), threshold_rows
    )
    write_csv(output_dir / "clusters.csv", list(cluster_rows[0]), cluster_rows)
    write_csv(
        output_dir / "cluster_members.csv",
        ["threshold", "cluster_id", "sample"],
        member_rows,
    )
    write_csv(output_dir / "cluster_size_summary.csv", list(size_rows[0]), size_rows)
    write_csv(
        output_dir / "zero_components.csv",
        [
            "threshold",
            "cluster_id",
            "zero_component_id",
            "component_size",
            "contradictory_pairs",
            "unverified_pairs",
            "verified_consistent_pairs",
            "minimum_direct_C",
            "median_direct_C",
        ],
        zero_rows,
    )
    write_csv(
        output_dir / "zero_component_members.csv",
        ["threshold", "cluster_id", "zero_component_id", "sample", "issue_types"],
        zero_member_rows,
    )
    write_csv(
        output_dir / "zero_issue_examples.csv",
        [
            "representative_rank",
            "threshold",
            "cluster_id",
            "zero_component_ordinal",
            "issue_type",
            "sample_a",
            "sample_b",
            "direct_D",
            "direct_C",
            "witness_path",
            "witness_edges",
        ],
        witness_rows,
    )
    write_csv(
        output_dir / "complete_case_pair_changes.csv",
        [
            "threshold",
            "cluster_id",
            "sample_a",
            "sample_b",
            "original_D",
            "original_C",
            "complete_case_D",
            "complete_case_C",
            "became_identical",
        ],
        change_rows,
    )
    write_csv(
        output_dir / "leave_one_out.csv",
        [
            "threshold",
            "cluster_id",
            "sample",
            "called_loci",
            "profile_completeness",
            "baseline_complete_case_loci",
            "loci_recovered_by_exclusion",
            "complete_case_loci_after_exclusion",
        ],
        leave_rows,
    )
    write_csv(
        output_dir / "representative_clusters.csv",
        [
            "rank",
            "threshold",
            "cluster_id",
            "cluster_size",
            "contradictory_pairs",
            "unverified_pairs",
            "profiles_file",
            "pairs_file",
        ],
        representative_rows,
    )

    summary_lines = [
        "cgMLST missing-data analysis",
        f"Samples: {len(cache.names)}",
        f"Loci: {locus_count}",
        f"Minimum comparable loci for an eligible pair: {required_c}",
        "",
    ]
    for row in threshold_rows:
        summary_lines.append(
            "Threshold {threshold}: {clusters} clusters; {problematic_clusters} "
            "problematic clusters; {contradictory_pairs} contradictory pairs; "
            "{unverified_pairs} unverified pairs".format(**row)
        )
    summary_lines.extend(
        [
            "",
            f"Representative problematic clusters exported: {len(representatives)}",
            "Contradictory means an observed direct difference within a zero-connected component.",
            "Unverified means the direct comparison does not meet the overlap requirement.",
        ]
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def parse_thresholds(value: str) -> list[int]:
    try:
        thresholds = sorted(
            {int(item.strip()) for item in value.split(",") if item.strip()}
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated integers"
        ) from error
    if not thresholds or thresholds[0] < 0:
        raise argparse.ArgumentTypeError(
            "at least one non-negative threshold is required"
        )
    return thresholds


def ensure_writable(paths: Sequence[Path], force: bool) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise UserError("output paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise UserError("output already exists (use --force): " + ", ".join(existing))


def command_compute(args: argparse.Namespace) -> None:
    outputs = [args.cache]
    if args.pairs is not None:
        outputs.append(args.pairs)
    metrics_path = Path(str(args.cache) + ".metrics.csv")
    outputs.append(metrics_path)
    pseudonymized_path = args.pseudonymized_output or args.cache.with_name(
        f"{args.cache.stem}.pseudonymized.tsv"
    )
    outputs.append(pseudonymized_path)
    if args.input.resolve() in {path.resolve() for path in outputs}:
        raise UserError("an output path cannot replace the input profile file")
    ensure_writable(outputs, args.force)
    metrics: list[dict] = []
    started = time.perf_counter()
    profiles = read_profiles(args.input, args.metadata_column)
    metrics.append(metric("input_encoding", started))

    profiles = pseudonymize_profiles(profiles, prefix=args.pseudonym_prefix)
    started = time.perf_counter()
    write_profile_subset(
        pseudonymized_path, profiles, tuple(range(len(profiles.names)))
    )
    metrics.append(metric("pseudonymized_profile_writing", started))

    started = time.perf_counter()
    distances, comparable = calculate_pairwise(profiles.profiles)
    metrics.append(metric("distance_matrix_calculation", started))
    cache = PairwiseCache(
        **profiles.__dict__, distances=distances, comparable=comparable
    )

    started = time.perf_counter()
    save_cache(args.cache, cache)
    metrics.append(metric("cache_writing", started))
    if args.pairs is not None:
        started = time.perf_counter()
        write_pair_csv(args.pairs, cache)
        metrics.append(metric("pair_csv_writing", started))
    write_metrics(metrics_path, metrics)
    print(f"Wrote cache: {args.cache}")
    print(f"Wrote pseudonymized profiles: {pseudonymized_path}")
    if args.pairs is not None:
        print(f"Wrote pair table: {args.pairs}")
    print(f"Wrote metrics: {metrics_path}")


def command_analyze(args: argparse.Namespace) -> None:
    generated_names = [
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
    ]
    cache_path = args.cache.resolve()
    generated_paths = {(args.output_dir / name).resolve() for name in generated_names}
    representative_path = (args.output_dir / "representative_clusters").resolve()
    if cache_path in generated_paths or cache_path.is_relative_to(representative_path):
        raise UserError("the cache path conflicts with generated analysis output")
    if args.output_dir.exists():
        if not args.force:
            raise UserError(
                f"output directory already exists (use --force): {args.output_dir}"
            )
        if not args.output_dir.is_dir():
            raise UserError(f"output path is not a directory: {args.output_dir}")

    metrics: list[dict] = []
    started = time.perf_counter()
    cache = load_cache(args.cache)
    metrics.append(metric("cache_loading", started))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        generated_targets = [args.output_dir / name for name in generated_names]
        for target in generated_targets:
            if target.is_dir() and not target.is_symlink():
                raise UserError(f"expected generated output to be a file: {target}")
        representative_dir = args.output_dir / "representative_clusters"
        if representative_dir.is_symlink() or (
            representative_dir.exists() and not representative_dir.is_dir()
        ):
            raise UserError(
                f"expected representative output to be a directory: {representative_dir}"
            )
        for target in generated_targets:
            target.unlink(missing_ok=True)
        if representative_dir.exists():
            shutil.rmtree(representative_dir)
    started = time.perf_counter()
    analyze_cache(
        cache=cache,
        thresholds=args.thresholds,
        output_dir=args.output_dir,
        minimum_loci=args.min_comparable_loci,
        minimum_proportion=args.min_comparable_proportion,
        representative_count=args.representative_count,
        max_witnesses=args.max_witnesses,
    )
    metrics.append(metric("clustering_complete_case_and_output", started))
    write_metrics(args.output_dir / "run_metrics.csv", metrics)
    print(f"Wrote analysis: {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze missing-data effects in categorical cgMLST profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Distances:
  D = different alleles at loci called in both samples
  C = loci called in both samples
  Pairs with C=0 are recorded but never used as clustering edges.

Typical workflow:
  cgmlst_missing_data.py compute profiles.tsv --metadata-column ST --cache profiles.npz
  cgmlst_missing_data.py analyze profiles.npz --thresholds 0,1,2,5 --output-dir results

Omit --min-comparable-loci and --min-comparable-proportion to apply no overlap
cutoff beyond requiring C>0. Tree generation is intentionally not included.
Repeated sample identifiers receive collision-safe -2, -3, ... suffixes.
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute = subparsers.add_parser(
        "compute", help="calculate reusable pairwise D and C arrays"
    )
    compute.add_argument("input", type=Path, help="GrapeTree-style profile table")
    compute.add_argument("--cache", required=True, type=Path, help="output .npz cache")
    compute.add_argument(
        "--metadata-column",
        action="append",
        default=[],
        help="non-locus column to preserve and exclude (repeatable)",
    )
    compute.add_argument(
        "--pairs", type=Path, help="optional complete .csv or .csv.gz pair table"
    )
    compute.add_argument(
        "--pseudonymized-output",
        type=Path,
        help="output profile copy (default: CACHE_STEM.pseudonymized.tsv)",
    )
    compute.add_argument(
        "--pseudonym-prefix",
        default="sample",
        help="prefix for consecutive identifiers (default: sample)",
    )
    compute.add_argument("--force", action="store_true", help="overwrite outputs")
    compute.set_defaults(func=command_compute)

    analyze = subparsers.add_parser(
        "analyze", help="cluster and analyze an existing pairwise cache"
    )
    analyze.add_argument("cache", type=Path, help="cache produced by compute")
    analyze.add_argument(
        "--thresholds",
        required=True,
        type=parse_thresholds,
        help="comma-separated D values",
    )
    analyze.add_argument("--output-dir", required=True, type=Path)
    overlap = analyze.add_mutually_exclusive_group()
    overlap.add_argument("--min-comparable-loci", type=int)
    overlap.add_argument("--min-comparable-proportion", type=float)
    analyze.add_argument("--representative-count", type=int, default=5)
    analyze.add_argument(
        "--max-witnesses",
        type=int,
        default=3,
        help="maximum examples per issue type and representative cluster",
    )
    analyze.add_argument(
        "--force", action="store_true", help="replace output directory"
    )
    analyze.set_defaults(func=command_analyze)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "compute":
        if args.cache.suffix != ".npz":
            raise UserError("--cache must end in .npz")
        if args.pairs is not None and not (
            args.pairs.name.endswith(".csv") or args.pairs.name.endswith(".csv.gz")
        ):
            raise UserError("--pairs must end in .csv or .csv.gz")
        if (
            not args.pseudonym_prefix
            or any(character.isspace() for character in args.pseudonym_prefix)
            or args.pseudonym_prefix == MISSING
        ):
            raise UserError("--pseudonym-prefix must be non-empty and contain no whitespace")
    else:
        if args.min_comparable_loci is not None and args.min_comparable_loci < 1:
            raise UserError("--min-comparable-loci must be at least 1")
        if args.min_comparable_proportion is not None and not (
            0 < args.min_comparable_proportion <= 1
        ):
            raise UserError(
                "--min-comparable-proportion must be greater than 0 and at most 1"
            )
        if args.representative_count < 0:
            raise UserError("--representative-count cannot be negative")
        if args.max_witnesses < 0:
            raise UserError("--max-witnesses cannot be negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        args.func(args)
    except UserError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
