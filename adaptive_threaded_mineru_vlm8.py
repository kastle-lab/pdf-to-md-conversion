#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# Imported later, after llama.cpp is patched.
base = None

# Each PDF thread gets its own FormulaCollector.
_THREAD_STATE = threading.local()


def _get_thread_collector():
    return getattr(_THREAD_STATE, "collector", None)


# ---------------------------------------------------------------------------
# llama.cpp n_parallel patch
# ---------------------------------------------------------------------------

def patch_llama_cpp_parallelism(n_parallel: int) -> str:
    """
    Force mineru-llama-cpp Engine to use the requested n_parallel.

    The installed default is currently 4.
    This wrapper defaults to 8.
    """
    if n_parallel < 1:
        raise ValueError("n_parallel must be at least 1")

    import mineru_llama_cpp

    original_engine = mineru_llama_cpp.Engine

    if getattr(original_engine, "_adaptive_parallel_patch", False):
        previous = getattr(
            original_engine,
            "_adaptive_n_parallel",
            None,
        )

        return (
            f"Engine already patched with "
            f"n_parallel={previous}"
        )

    if not inspect.isclass(original_engine):
        raise TypeError(
            "mineru_llama_cpp.Engine is not a class"
        )

    # Keep Engine as a class so isinstance() checks still work.
    class ParallelEngine(original_engine):

        def __init__(self, *args, **kwargs):
            kwargs["n_parallel"] = n_parallel
            super().__init__(*args, **kwargs)

    ParallelEngine.__name__ = original_engine.__name__
    ParallelEngine.__qualname__ = original_engine.__qualname__
    ParallelEngine.__module__ = original_engine.__module__

    ParallelEngine._adaptive_parallel_patch = True
    ParallelEngine._adaptive_n_parallel = n_parallel
    ParallelEngine._original_engine = original_engine

    mineru_llama_cpp.Engine = ParallelEngine

    # If any MinerU modules already imported the old Engine class,
    # replace those aliases too.
    for module_name, module in list(sys.modules.items()):

        if module is None:
            continue

        if not module_name.startswith("mineru"):
            continue

        try:
            if (
                getattr(module, "Engine", None)
                is original_engine
            ):
                setattr(
                    module,
                    "Engine",
                    ParallelEngine,
                )
        except Exception:
            pass

    return (
        "mineru_llama_cpp.Engine forced to "
        f"n_parallel={n_parallel}"
    )


# ---------------------------------------------------------------------------
# Thread-safe formula collector patch
# ---------------------------------------------------------------------------

def patch_formula_recognizers_threadsafe() -> list[str]:
    """
    Patch MinerU's formula-recognition batch_predict methods.

    The difference from the original pipeline is that the FormulaCollector
    is stored per thread instead of using one shared _ACTIVE_COLLECTOR.
    """

    module_names = [
        "mineru.model.mfr.pp_formulanet.predict_formula",
        "mineru.model.mfr.pp_formulanet_plus_m_onnx",
        "mineru.model.mfr.unimernet.Unimernet",
        "mineru.model.mfr.pp_formulanet_plus_m.predict_formula",
    ]

    patched: list[str] = []

    for module_name in module_names:

        try:
            module = importlib.import_module(
                module_name
            )
        except Exception:
            continue

        for class_name, cls in inspect.getmembers(
            module,
            inspect.isclass,
        ):

            if (
                getattr(cls, "__module__", None)
                != module.__name__
            ):
                continue

            original = getattr(
                cls,
                "batch_predict",
                None,
            )

            if not callable(original):
                continue

            # Prevent double wrapping.
            if getattr(
                original,
                "_threaded_mineru_wrapper",
                False,
            ):
                continue

            try:
                source = inspect.getsource(
                    original
                )
            except Exception:
                source = ""

            # The formula recognizers used by MinerU contain this text.
            if "MFR Predict" not in source:
                continue

            source_name = (
                f"{module.__name__}."
                f"{class_name}.batch_predict"
            )

            def make_wrapper(
                orig,
                src_name,
            ):

                def wrapped(
                    self,
                    *args,
                    **kwargs,
                ):

                    # First let MinerU perform its normal
                    # formula recognition.
                    result = orig(
                        self,
                        *args,
                        **kwargs,
                    )

                    # Get collector belonging ONLY
                    # to this PDF thread.
                    collector = (
                        _get_thread_collector()
                    )

                    if collector is None:
                        return result

                    # Try to identify detections.
                    if len(args) >= 1:
                        detections = args[0]

                    else:
                        detections = kwargs.get(
                            "images_mfd_res"
                        )

                        if detections is None:
                            detections = kwargs.get(
                                "mfd_res"
                            )

                    # Try to identify images.
                    if len(args) >= 2:
                        images = args[1]

                    else:
                        images = kwargs.get(
                            "images"
                        )

                        if images is None:
                            images = kwargs.get(
                                "image"
                            )

                    if isinstance(
                        images,
                        np.ndarray,
                    ):
                        images = [images]

                    if (
                        isinstance(
                            detections,
                            list,
                        )
                        and detections
                        and isinstance(
                            detections[0],
                            dict,
                        )
                        and isinstance(
                            images,
                            (list, tuple),
                        )
                        and len(images) == 1
                    ):
                        detections = [
                            detections
                        ]

                    if (
                        detections is None
                        or images is None
                    ):

                        collector.errors.append(
                            f"{src_name}: "
                            "could not identify "
                            "detections/images"
                        )

                    else:

                        collector.capture(
                            src_name,
                            detections,
                            images,
                            result,
                        )

                    return result

                wrapped._threaded_mineru_wrapper = True
                wrapped._threaded_original = orig

                return wrapped

            setattr(
                cls,
                "batch_predict",
                make_wrapper(
                    original,
                    source_name,
                ),
            )

            patched.append(
                source_name
            )

    if not patched:
        raise RuntimeError(
            "No FormulaRecognizer "
            "batch_predict() implementation "
            "was patched."
        )

    # Keep the original debug report accurate.
    if hasattr(base, "_PATCHED"):

        base._PATCHED.clear()

        base._PATCHED.extend(
            patched
        )

    return patched


# ---------------------------------------------------------------------------
# Process one PDF
# ---------------------------------------------------------------------------

def process_one_pdf(
    pdf: Path,
    output_dir: Path,
    tier: str,
    ocr_mode: str,
    remove_aside_text: bool,
    parse_fn,
    writer_cls,
) -> dict[str, Any]:

    started = time.perf_counter()

    paper_out = (
        output_dir
        / pdf.stem
    )

    paper_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    collector = base.FormulaCollector(
        paper_out
    )

    # Collector belongs only to this thread.
    _THREAD_STATE.collector = collector

    try:

        result = parse_fn(
            pdf,
            tier=tier,
            ocr_mode=ocr_mode,
            page_range="all",
        )

        # Formula-recognition stage is done.
        _THREAD_STATE.collector = None

        result.save(
            writer_cls(
                str(paper_out)
            )
        )

        collector.save()

        cleanup_report = (
            base.clean_markdown_from_labeled_noise(
                paper_out,
                remove_aside_text=(
                    remove_aside_text
                ),
            )
        )

        footnote_report = (
            base.remove_noisy_footnotes_from_markdown(
                paper_out
            )
        )

        middle = (
            paper_out
            / "middle_json.json"
        )

        layout_ok = False

        layout_message = (
            "middle_json.json not found"
        )

        if middle.exists():

            layout_ok, layout_message = (
                base.create_layout_pdf(
                    pdf,
                    middle,
                    paper_out
                    / f"{pdf.stem}_layout.pdf",
                )
            )

        elapsed = (
            time.perf_counter()
            - started
        )

        return {
            "ok": True,

            "pdf": pdf.name,

            "pid": os.getpid(),

            "thread_id": (
                threading.get_ident()
            ),

            "elapsed_seconds": round(
                elapsed,
                3,
            ),

            "layout_pdf": (
                layout_ok
            ),

            "layout_message": (
                layout_message
            ),

            "recognizer_hook_calls": (
                len(collector.calls)
            ),

            "equation_images": (
                len(collector.items)
            ),

            "markdown_noise_removed": (
                cleanup_report.get(
                    "lines_removed",
                    0,
                )
            ),

            "footnotes_kept": (
                footnote_report.get(
                    "kept",
                    0,
                )
            ),

            "footnotes_noise": (
                footnote_report.get(
                    "removed",
                    0,
                )
            ),

            "footnotes_removed": (
                footnote_report.get(
                    "matched_and_removed_from_markdown",
                    0,
                )
            ),

            "capture_warnings": (
                list(
                    collector.errors
                )
            ),
        }

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - started
        )

        return {
            "ok": False,

            "pdf": pdf.name,

            "pid": os.getpid(),

            "thread_id": (
                threading.get_ident()
            ),

            "elapsed_seconds": round(
                elapsed,
                3,
            ),

            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),

            "capture_warnings": (
                list(
                    collector.errors
                )
            ),
        }

    finally:

        # ThreadPoolExecutor reuses its threads.
        # Never leave the previous PDF's collector attached.
        try:
            del _THREAD_STATE.collector

        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Print completed PDF result
# ---------------------------------------------------------------------------

def print_result(
    result: dict[str, Any],
    completed: int,
    total: int,
) -> None:

    status = (
        "OK"
        if result.get("ok")
        else "FAILED"
    )

    print(
        f"[{completed}/{total}] "
        f"{result.get('pdf')} "
        f"— {status} — "
        f"{result.get('elapsed_seconds', 0):.1f}s "
        f"(PID {result.get('pid')}, "
        f"thread {result.get('thread_id')})"
    )

    if not result.get("ok"):

        print(
            "    ERROR:",
            result.get("error"),
        )

        return

    print(
        "    layout PDF:",
        (
            "yes"
            if result.get("layout_pdf")
            else "NO"
        ),
    )

    print(
        "    recognizer hook calls:",
        result.get(
            "recognizer_hook_calls",
            0,
        ),
    )

    print(
        "    equation images:",
        result.get(
            "equation_images",
            0,
        ),
    )

    print(
        "    Markdown noise removed:",
        result.get(
            "markdown_noise_removed",
            0,
        ),
        "line(s)",
    )

    print(
        "    footnotes:",
        result.get(
            "footnotes_kept",
            0,
        ),
        "kept,",
        result.get(
            "footnotes_noise",
            0,
        ),
        "classified as noise,",
        result.get(
            "footnotes_removed",
            0,
        ),
        "removed from Markdown",
    )

    for warning in result.get(
        "capture_warnings",
        [],
    ):

        print(
            "    CAPTURE WARNING:",
            warning,
        )


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------

class ResourceMonitor:

    def __init__(
        self,
        gpu_index: int = 0,
    ):

        self.gpu_index = gpu_index

        self._last_cpu = None


    @staticmethod
    def _read_cpu_times():

        with open(
            "/proc/stat",
            "r",
            encoding="utf-8",
        ) as f:

            fields = (
                f.readline()
                .split()
            )

        if (
            not fields
            or fields[0] != "cpu"
        ):

            raise RuntimeError(
                "Could not read "
                "aggregate CPU statistics"
            )

        values = [
            int(x)
            for x in fields[1:]
        ]

        while len(values) < 8:
            values.append(0)

        (
            user,
            nice,
            system,
            idle,
            iowait,
            irq,
            softirq,
            steal,
        ) = values[:8]

        idle_all = (
            idle
            + iowait
        )

        non_idle = (
            user
            + nice
            + system
            + irq
            + softirq
            + steal
        )

        total = (
            idle_all
            + non_idle
        )

        return (
            total,
            idle_all,
        )


    def cpu_percent(
        self,
    ) -> float:

        current = (
            self._read_cpu_times()
        )

        if self._last_cpu is None:

            self._last_cpu = current

            time.sleep(
                0.15
            )

            current = (
                self._read_cpu_times()
            )

        prev_total, prev_idle = (
            self._last_cpu
        )

        total, idle = current

        self._last_cpu = current

        total_delta = (
            total
            - prev_total
        )

        idle_delta = (
            idle
            - prev_idle
        )

        if total_delta <= 0:
            return 0.0

        busy = (
            total_delta
            - idle_delta
        )

        return max(
            0.0,
            min(
                100.0,
                busy
                * 100.0
                / total_delta,
            ),
        )


    @staticmethod
    def available_ram_gb() -> float:

        with open(
            "/proc/meminfo",
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.startswith(
                    "MemAvailable:"
                ):

                    kb = int(
                        line.split()[1]
                    )

                    return (
                        kb
                        / 1024
                        / 1024
                    )

        raise RuntimeError(
            "Could not read "
            "MemAvailable from "
            "/proc/meminfo"
        )


    def gpu_stats(
        self,
    ) -> dict[str, float]:

        cmd = [
            "nvidia-smi",

            "-i",
            str(
                self.gpu_index
            ),

            "--query-gpu="
            "memory.free,"
            "memory.total,"
            "utilization.gpu",

            "--format="
            "csv,noheader,nounits",
        ]

        output = (
            subprocess.check_output(
                cmd,
                text=True,
                stderr=(
                    subprocess.STDOUT
                ),
            )
            .strip()
        )

        parts = [
            p.strip()
            for p in (
                output
                .splitlines()[0]
                .split(",")
            )
        ]

        if len(parts) < 3:

            raise RuntimeError(
                "Unexpected "
                "nvidia-smi output: "
                f"{output}"
            )

        return {
            "vram_free_mb": float(
                parts[0]
            ),

            "vram_total_mb": float(
                parts[1]
            ),

            "gpu_util_percent": float(
                parts[2]
            ),
        }


    def snapshot(
        self,
    ) -> dict[str, float]:

        gpu = (
            self.gpu_stats()
        )

        return {
            "cpu_percent": (
                self.cpu_percent()
            ),

            "ram_available_gb": (
                self.available_ram_gb()
            ),

            **gpu,
        }


# ---------------------------------------------------------------------------
# Decide whether another PDF should launch
# ---------------------------------------------------------------------------

def resources_allow_launch(
    snapshot: dict[str, float],
    max_cpu_percent: float,
    min_free_ram_gb: float,
    min_free_vram_mb: float,
) -> tuple[bool, list[str]]:

    reasons: list[str] = []

    if (
        snapshot["cpu_percent"]
        >= max_cpu_percent
    ):

        reasons.append(
            f"CPU "
            f"{snapshot['cpu_percent']:.1f}% "
            f">= "
            f"{max_cpu_percent:.1f}%"
        )

    if (
        snapshot["ram_available_gb"]
        < min_free_ram_gb
    ):

        reasons.append(
            f"free RAM "
            f"{snapshot['ram_available_gb']:.1f} GB "
            f"< "
            f"{min_free_ram_gb:.1f} GB"
        )

    if (
        snapshot["vram_free_mb"]
        < min_free_vram_mb
    ):

        reasons.append(
            f"free VRAM "
            f"{snapshot['vram_free_mb']:.0f} MB "
            f"< "
            f"{min_free_vram_mb:.0f} MB"
        )

    return (
        not reasons,
        reasons,
    )


def format_resources(
    snapshot: dict[str, float],
) -> str:

    return (
        f"CPU "
        f"{snapshot['cpu_percent']:.1f}% | "

        f"RAM free "
        f"{snapshot['ram_available_gb']:.1f} GB | "

        f"VRAM free "
        f"{snapshot['vram_free_mb']:.0f}/"
        f"{snapshot['vram_total_mb']:.0f} MB | "

        f"GPU "
        f"{snapshot['gpu_util_percent']:.0f}%"
    )


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def format_duration(
    seconds: float,
) -> str:

    seconds = max(
        0,
        int(
            round(seconds)
        ),
    )

    days, rem = divmod(
        seconds,
        86400,
    )

    hours, rem = divmod(
        rem,
        3600,
    )

    minutes, secs = divmod(
        rem,
        60,
    )

    if days:

        return (
            f"{days}d "
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Adaptive multithreaded MinerU "
            "wrapper with increased llama.cpp "
            "VLM parallelism."
        )
    )

    parser.add_argument(
        "input_dir",
        type=Path,
    )

    parser.add_argument(
        "output_dir",
        type=Path,
    )

    parser.add_argument(
        "--tier",
        default="standard",
        choices=[
            "flash",
            "basic",
            "standard",
            "advanced",
        ],
    )

    parser.add_argument(
        "--ocr-mode",
        default="auto",
        choices=[
            "auto",
            "on",
            "off",
        ],
    )

    parser.add_argument(
        "--remove-aside-text",
        action="store_true",
    )

    # MinerU's installed default is 4.
    # This wrapper starts with 8.
    parser.add_argument(
        "--vlm-parallel",
        type=int,
        default=8,
        help=(
            "llama.cpp concurrent VLM slots. "
            "Default: 8."
        ),
    )

    # Prevent the previous runaway-thread problem.
    parser.add_argument(
        "--max-active-pdfs",
        type=int,
        default=8,
        help=(
            "Hard maximum number of PDFs "
            "running at the same time. "
            "Default: 8."
        ),
    )

    parser.add_argument(
        "--max-cpu",
        type=float,
        default=90.0,
        help=(
            "Do not start another PDF "
            "at or above this total "
            "CPU usage."
        ),
    )

    parser.add_argument(
        "--min-free-ram-gb",
        type=float,
        default=16.0,
        help=(
            "Minimum available RAM "
            "required before starting "
            "another PDF."
        ),
    )

    parser.add_argument(
        "--min-free-vram-mb",
        type=float,
        default=2500.0,
        help=(
            "Minimum free GPU VRAM "
            "required before starting "
            "another PDF."
        ),
    )

    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help=(
            "NVIDIA GPU index used "
            "for monitoring."
        ),
    )

    parser.add_argument(
        "--launch-delay",
        type=float,
        default=5.0,
        help=(
            "Minimum seconds between "
            "new PDF launches."
        ),
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help=(
            "Scheduler polling interval "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--status-interval",
        type=float,
        default=10.0,
        help=(
            "Seconds between WAITING "
            "messages."
        ),
    )

    args = parser.parse_args()


    # -------------------------------------------------------
    # Validate arguments
    # -------------------------------------------------------

    if args.vlm_parallel < 1:

        raise SystemExit(
            "--vlm-parallel "
            "must be at least 1"
        )

    if args.max_active_pdfs < 1:

        raise SystemExit(
            "--max-active-pdfs "
            "must be at least 1"
        )

    if not (
        0
        < args.max_cpu
        <= 100
    ):

        raise SystemExit(
            "--max-cpu must be "
            "between 0 and 100"
        )

    if (
        args.min_free_ram_gb
        < 0
    ):

        raise SystemExit(
            "--min-free-ram-gb "
            "cannot be negative"
        )

    if (
        args.min_free_vram_mb
        < 0
    ):

        raise SystemExit(
            "--min-free-vram-mb "
            "cannot be negative"
        )

    if (
        args.launch_delay
        < 0
    ):

        raise SystemExit(
            "--launch-delay "
            "cannot be negative"
        )

    if (
        args.poll_interval
        <= 0
    ):

        raise SystemExit(
            "--poll-interval "
            "must be greater than 0"
        )


    # -------------------------------------------------------
    # Find PDFs
    # -------------------------------------------------------

    input_dir = (
        args.input_dir
        .resolve()
    )

    output_dir = (
        args.output_dir
        .resolve()
    )

    pdfs = sorted(
        p
        for p in input_dir.rglob(
            "*.pdf"
        )
        if p.is_file()
    )

    if not pdfs:

        raise SystemExit(
            "No PDFs found"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------
    # Patch llama.cpp BEFORE importing your existing
    # batch_mineru_research pipeline.
    # -------------------------------------------------------

    vlm_patch = (
        patch_llama_cpp_parallelism(
            args.vlm_parallel
        )
    )


    # -------------------------------------------------------
    # Import existing pipeline only after llama.cpp patch
    # -------------------------------------------------------

    global base

    import batch_mineru_research as imported_base

    base = imported_base


    # -------------------------------------------------------
    # Install thread-safe formula patch
    # -------------------------------------------------------

    patched = (
        patch_formula_recognizers_threadsafe()
    )


    # -------------------------------------------------------
    # Import MinerU parser
    # -------------------------------------------------------

    from mineru.parser import parse

    from mineru.parser.writer import (
        FileBasedDataWriter
    )


    monitor = ResourceMonitor(
        args.gpu_index
    )


    # Hard safety ceiling.
    executor_capacity = min(
        args.max_active_pdfs,
        len(pdfs),
    )


    # -------------------------------------------------------
    # Startup information
    # -------------------------------------------------------

    print(
        f"PDFs found: "
        f"{len(pdfs)}"
    )

    print(
        f"Python process PID: "
        f"{os.getpid()}"
    )

    print(
        "Execution model: "
        "one process + "
        "adaptive ThreadPoolExecutor"
    )

    print(
        "Collector state: "
        "thread-local"
    )

    print(
        "llama.cpp VLM "
        f"n_parallel: "
        f"{args.vlm_parallel}"
    )

    print(
        "Maximum active PDFs: "
        f"{executor_capacity}"
    )

    print(
        "VLM patch:",
        vlm_patch,
    )

    print()

    print(
        "Adaptive launch rules:"
    )

    print(
        "    CPU below: "
        f"{args.max_cpu:.1f}%"
    )

    print(
        "    free RAM at least: "
        f"{args.min_free_ram_gb:.1f} GB"
    )

    print(
        "    free VRAM at least: "
        f"{args.min_free_vram_mb:.0f} MB"
    )

    print(
        "    launch delay: "
        f"{args.launch_delay:.1f}s"
    )

    print()

    print(
        "Patched MFR method(s):"
    )

    for method in patched:

        print(
            "   ",
            method,
        )

    print()


    # -------------------------------------------------------
    # Start timing
    # -------------------------------------------------------

    batch_started_wall = (
        datetime.now(
            timezone.utc
        )
    )

    batch_started = (
        time.perf_counter()
    )


    # Final report.
    summary_path = (
        output_dir
        / "batch_summary.json"
    )


    # Each completed PDF is written immediately.
    results_jsonl_path = (
        output_dir
        / "pdf_results.jsonl"
    )


    # Clear old per-PDF result file for this run.
    results_jsonl_path.write_text(
        "",
        encoding="utf-8",
    )


    pending = deque(
        pdfs
    )

    running: dict[
        Any,
        Path,
    ] = {}


    completed = 0

    succeeded = 0

    failed = 0

    peak_active_pdfs = 0

    next_launch_time = 0.0

    last_status_time = 0.0


    # -------------------------------------------------------
    # Thread pool
    # -------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=executor_capacity,
        thread_name_prefix="mineru-pdf",
    ) as executor:


        while (
            pending
            or running
        ):


            # ------------------------------------------------
            # Check for finished PDFs
            # ------------------------------------------------

            for future in list(
                running
            ):


                if not future.done():

                    continue


                pdf = running.pop(
                    future
                )


                completed += 1


                try:

                    result = (
                        future.result()
                    )


                except Exception as exc:

                    result = {
                        "ok": False,

                        "pdf": (
                            pdf.name
                        ),

                        "pid": (
                            os.getpid()
                        ),

                        "thread_id": None,

                        "elapsed_seconds": 0.0,

                        "error": (
                            "Thread wrapper failure: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }


                if result.get(
                    "ok"
                ):

                    succeeded += 1

                else:

                    failed += 1


                # Save individual result immediately.
                append_jsonl(
                    results_jsonl_path,
                    result,
                )


                print_result(
                    result,
                    completed,
                    len(pdfs),
                )


            # ------------------------------------------------
            # Decide whether another PDF can start
            # ------------------------------------------------

            now = (
                time.monotonic()
            )


            if (
                pending
                and len(running)
                < executor_capacity
                and now
                >= next_launch_time
            ):


                try:

                    snapshot = (
                        monitor.snapshot()
                    )


                    allowed, reasons = (
                        resources_allow_launch(
                            snapshot,

                            args.max_cpu,

                            args.min_free_ram_gb,

                            args.min_free_vram_mb,
                        )
                    )


                except Exception as exc:

                    allowed = False

                    reasons = [
                        "resource monitor error: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ]

                    snapshot = None


                # --------------------------------------------
                # Launch next PDF
                # --------------------------------------------

                if allowed:


                    pdf = (
                        pending.popleft()
                    )


                    future = executor.submit(
                        process_one_pdf,

                        pdf,

                        output_dir,

                        args.tier,

                        args.ocr_mode,

                        args.remove_aside_text,

                        parse,

                        FileBasedDataWriter,
                    )


                    running[
                        future
                    ] = pdf


                    peak_active_pdfs = max(
                        peak_active_pdfs,
                        len(running),
                    )


                    next_launch_time = (
                        now
                        + args.launch_delay
                    )


                    print(
                        f"[LAUNCH] "
                        f"{pdf.name} | "

                        f"active PDFs: "
                        f"{len(running)} | "

                        f"waiting: "
                        f"{len(pending)} | "

                        f"{format_resources(snapshot)}"
                    )


                # --------------------------------------------
                # Resources currently too busy
                # --------------------------------------------

                elif (
                    now
                    - last_status_time
                    >= args.status_interval
                ):


                    last_status_time = (
                        now
                    )


                    if (
                        snapshot
                        is not None
                    ):

                        resource_text = (
                            format_resources(
                                snapshot
                            )
                        )

                    else:

                        resource_text = (
                            "resource snapshot "
                            "unavailable"
                        )


                    print(
                        f"[WAITING] "

                        f"active PDFs: "
                        f"{len(running)} | "

                        f"waiting: "
                        f"{len(pending)} | "

                        f"{resource_text}"
                    )


                    print(
                        "          reason:",
                        "; ".join(
                            reasons
                        ),
                    )


            # Avoid a busy scheduler loop.
            if (
                pending
                or running
            ):

                time.sleep(
                    args.poll_interval
                )


    # -------------------------------------------------------
    # Finished
    # -------------------------------------------------------

    batch_elapsed = (
        time.perf_counter()
        - batch_started
    )


    batch_finished_wall = (
        datetime.now(
            timezone.utc
        )
    )


    if (
        batch_elapsed
        > 0
    ):

        throughput = (
            succeeded
            / batch_elapsed
        ) * 60.0

    else:

        throughput = 0.0


    # -------------------------------------------------------
    # Final JSON report
    # -------------------------------------------------------

    summary = {

        "status": (
            "completed"
        ),

        "started_at_utc": (
            batch_started_wall
            .isoformat()
        ),

        "finished_at_utc": (
            batch_finished_wall
            .isoformat()
        ),

        "total_wall_time_seconds": round(
            batch_elapsed,
            3,
        ),

        "total_wall_time_human": (
            format_duration(
                batch_elapsed
            )
        ),

        "pdfs_total": (
            len(pdfs)
        ),

        "succeeded": (
            succeeded
        ),

        "failed": (
            failed
        ),

        "successful_throughput_pdf_per_min": round(
            throughput,
            4,
        ),

        "peak_active_pdfs": (
            peak_active_pdfs
        ),

        "configuration": {

            "tier": (
                args.tier
            ),

            "ocr_mode": (
                args.ocr_mode
            ),

            "vlm_parallel": (
                args.vlm_parallel
            ),

            "max_active_pdfs": (
                executor_capacity
            ),

            "max_cpu_percent": (
                args.max_cpu
            ),

            "min_free_ram_gb": (
                args.min_free_ram_gb
            ),

            "min_free_vram_mb": (
                args.min_free_vram_mb
            ),

            "launch_delay_seconds": (
                args.launch_delay
            ),

            "poll_interval_seconds": (
                args.poll_interval
            ),

            "gpu_index": (
                args.gpu_index
            ),
        },

        "per_pdf_results_file": (
            str(
                results_jsonl_path
            )
        ),
    }


    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),

        encoding="utf-8",
    )


    # -------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "BATCH COMPLETE"
    )

    print(
        f"    PDFs: "
        f"{len(pdfs)}"
    )

    print(
        f"    succeeded: "
        f"{succeeded}"
    )

    print(
        f"    failed: "
        f"{failed}"
    )

    print(
        f"    VLM n_parallel: "
        f"{args.vlm_parallel}"
    )

    print(
        f"    max active PDFs: "
        f"{executor_capacity}"
    )

    print(
        f"    peak active PDFs: "
        f"{peak_active_pdfs}"
    )

    print(
        f"    total wall time: "
        f"{batch_elapsed:.1f}s"
    )

    print(
        f"    total wall time: "
        f"{format_duration(batch_elapsed)}"
    )

    print(
        f"    successful throughput: "
        f"{throughput:.2f} PDF/min"
    )

    print(
        f"    summary report: "
        f"{summary_path}"
    )

    print(
        f"    per-PDF results: "
        f"{results_jsonl_path}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()