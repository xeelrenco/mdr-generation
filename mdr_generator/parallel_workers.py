"""Thread-pool helpers for I/O-bound LLM calls with live console progress."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, TypeVar

from .config import cfg_int
from .utils import format_elapsed_seconds

T = TypeVar("T")
R = TypeVar("R")

_log_lock = threading.Lock()


def pipeline_log(message: str) -> None:
    with _log_lock:
        print(message, flush=True)


def llm_parallel_workers() -> int:
    return max(1, cfg_int("LLM_PARALLEL_WORKERS", 8))


def run_parallel(
    items: List[T],
    fn: Callable[[T], R],
    *,
    max_workers: Optional[int] = None,
    label: str = "LLM",
    describe: Optional[Callable[[T], str]] = None,
    result_note: Optional[Callable[[T, R], str]] = None,
) -> List[R]:
    """Run fn on each item; log progress as tasks complete; preserve input order."""
    if not items:
        return []

    total = len(items)
    workers = max(1, max_workers or llm_parallel_workers())
    cap = min(workers, total)

    def _desc(item: T, index: int) -> str:
        if describe:
            return describe(item)
        return f"task {index + 1}/{total}"

    def _note(item: T, result: R) -> str:
        if not result_note:
            return ""
        suffix = result_note(item, result).strip()
        return f" {suffix}" if suffix else ""

    if cap == 1:
        pipeline_log(f"  [{label}] avvio {total} task (sequenziale)")
        batch_start = time.perf_counter()
        results: List[R] = []
        for index, item in enumerate(items):
            task_start = time.perf_counter()
            pipeline_log(f"  [{label}] START {_desc(item, index)}")
            result = fn(item)
            task_elapsed = time.perf_counter() - task_start
            results.append(result)
            pipeline_log(
                f"  [{label}] {index + 1}/{total} DONE {_desc(item, index)}"
                f"{_note(item, result)} ({format_elapsed_seconds(task_elapsed)})"
            )
        pipeline_log(
            f"  [{label}] completato {total}/{total} "
            f"({format_elapsed_seconds(time.perf_counter() - batch_start)})"
        )
        return results

    pipeline_log(f"  [{label}] avvio {total} task, workers={cap}")
    batch_start = time.perf_counter()
    ordered: List[Optional[R]] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=cap) as pool:
        future_map = {
            pool.submit(fn, item): index for index, item in enumerate(items)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            item = items[index]
            result = future.result()
            ordered[index] = result
            completed += 1
            elapsed = time.perf_counter() - batch_start
            eta = ""
            if completed >= 2 and completed < total:
                avg = elapsed / completed
                eta = f", ETA ~{format_elapsed_seconds(avg * (total - completed))}"
            pipeline_log(
                f"  [{label}] {completed}/{total} DONE {_desc(item, index)}"
                f"{_note(item, result)} "
                f"({format_elapsed_seconds(elapsed)} batch{eta})"
            )

    pipeline_log(
        f"  [{label}] completato {total}/{total} "
        f"({format_elapsed_seconds(time.perf_counter() - batch_start)})"
    )
    return ordered  # type: ignore[return-value]
