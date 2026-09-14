"""QSA compiled program identity does not depend on the cache pool's page counts.

The number of main and compressed cache pages sizes the caller's pool and
changes between starts whenever the available memory differs. No QSA program
takes a page count as a compile-time constant or a launch argument, so two
declarations that differ only in those counts plan the same compiled programs
and a cached restart with a differently sized pool recompiles nothing.

Compile planning runs in a spawned offline compiler worker, the same process
shape the compile pool uses, so the host test needs no GPU and leaves the
test process's CUDA state untouched.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import pathlib

import pytest

CORPUS = pathlib.Path(__file__).parent.parent / "preparation" / "corpora"
CORPUS_FILE = CORPUS / "qwen3_8_flash_next_tp2_rank0.jsonl"
DEVICE_UUID = "48d28d14-08f3-f3d1-cb99-20c3fa5eca41"
PRODUCT_NAME = "NVIDIA GeForce RTX 5090"

# Page counts on both sides of Triton's integer specialization boundaries: a
# multiple of 16 and its neighbour; the smallest legal pool of each query joins them.
PAGE_COUNTS = ((2, 2), (16, 16), (21536, 5384), (21537, 5385))


def _smallest_legal_page_counts(payload: dict[str, object]) -> tuple[int, int]:
    max_seq_len = int(payload["max_seq_len"])
    groups = -(-max_seq_len // int(payload["compress_ratio"]))
    return (
        -(-max_seq_len // int(payload["main_page_size"])),
        -(-groups // int(payload["compressed_page_size"])),
    )


def _qsa_corpus_queries() -> list[dict[str, object]]:
    queries, seen = [], set()
    for line in CORPUS_FILE.read_text().splitlines():
        record = json.loads(line)
        if record["component"] != "attention.qsa" or record["request"] in seen:
            continue
        seen.add(record["request"])
        queries.append(record["query"])
    return queries


def _program_keys_by_geometry(payload: dict[str, object], connection) -> None:
    """Report the planned program keys of each page-count variant of one query."""
    try:
        from b12x._lib import compile_pool
        from b12x._lib.compile_pool import CompileJob, describe_compilation
        from b12x.attention.qsa._tuning import TUNING, QsaQuery
        from b12x.preparation import DeviceIdentity

        activity = multiprocessing.get_context("spawn").Array("q", (0, 0))
        compile_pool._initialize_worker(
            0, (12, 0), DEVICE_UUID, PRODUCT_NAME, 170, 232448, 232448, activity,
        )
        identity = DeviceIdentity("nvidia", (12, 0), 170, PRODUCT_NAME)
        keys = {}
        for main_pages, compressed_pages in (
            *PAGE_COUNTS, _smallest_legal_page_counts(payload),
        ):
            query = QsaQuery(**{
                **payload,
                "num_main_cache_pages": main_pages,
                "num_compressed_cache_pages": compressed_pages,
            })
            config = TUNING.default_config(query, identity)
            job = CompileJob.create(
                "b12x.attention.qsa._contract:compile_qsa",
                query.to_dict(), TUNING.encode_config(config), 0,
            )
            plan = describe_compilation(job)
            keys[(main_pages, compressed_pages)] = sorted(
                (program.dialect, program.key, program.name) for program in plan.programs
            )
        connection.send(("ok", keys))
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _plan_in_offline_worker(payload: dict[str, object], cache_dir: pathlib.Path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "B12X_COMPILE_CACHE_DIR": str(cache_dir / "compile"),
        "TRITON_CACHE_DIR": str(cache_dir / "triton"),
    }
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    try:
        process = context.Process(target=_program_keys_by_geometry, args=(payload, child))
        process.start()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    child.close()
    try:
        status, result = parent.recv()
    finally:
        process.join(timeout=600)
    if status != "ok":
        raise AssertionError(f"offline QSA compile planning failed: {result}")
    return result


@pytest.mark.parametrize(
    "payload", _qsa_corpus_queries(),
    ids=lambda payload: f"m{payload['max_q_rows']}-context{payload['max_seq_len']}",
)
def test_qsa_program_keys_ignore_the_cache_page_counts(payload, tmp_path) -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("triton")
    keys = _plan_in_offline_worker(payload, tmp_path)
    reference_geometry, reference = next(iter(keys.items()))
    assert reference, "QSA compile planning reported no programs"
    for geometry, programs in keys.items():
        assert programs == reference, (
            f"QSA programs for page counts {geometry} differ from {reference_geometry}: "
            f"{sorted(set(map(tuple, programs)) ^ set(map(tuple, reference)))}"
        )
