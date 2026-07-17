#!/usr/bin/env python3

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kgw_distilled import resolve_distilled_model  # noqa: E402


def _run_command(cmd: List[str], env: Dict[str, str]) -> Tuple[int, str, str]:
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def _read_selector_metrics(selector_matrix_dir: str) -> Dict[str, Any]:
    selector_metrics_path = os.path.join(selector_matrix_dir.rstrip("/"), "selector_metrics.json")
    if not os.path.isfile(selector_metrics_path):
        return {}
    with open(selector_metrics_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _kwargs_to_cli_args(kwargs: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in kwargs.items():
        if value is None:
            continue
        args.extend([f"--{key}", str(value)])
    return args


def _build_output_file(
    output_dir: str,
    method: str,
    param: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    context: Dict[str, Any] = {}
    output_file = ""
    dataset = str(param["dataset"])
    watermark_seed = int(param["watermark_seed"])

    if method == "gaussmark":
        sigma = param["sigma"]

        output_file = (
            f"{output_dir}/output_seed={watermark_seed}_sigma={sigma}_watermark={method}_dataset={dataset}"
        )
    elif method in {"unremovable", "christ"}:
        epsilon = param["epsilon"]
        # Canonical method id for new runs; keep "christ" alias for legacy configs.
        method_id = "unremovable" if method == "unremovable" else method
        output_file = (
            f"{output_dir}/output_seed={watermark_seed}_epsilon={epsilon}_watermark={method_id}_dataset={dataset}"
        )
    elif method in {"openstamp", "openstamp_binom", "openstamp_discrete"}:
        selector_matrix_dir = param["selector_matrix_dir"]
        delta = param["delta"]
        gamma = param["gamma"]
        metrics = _read_selector_metrics(selector_matrix_dir)
        sem_align = metrics.get("sem_align", False)
        k = metrics["k"]

        suffix = ""

        if sem_align:
            embedding_model = metrics["embedding_model"]
            align_method = metrics["align_method"]
            suffix = f"_semalign_{align_method}_embedding={embedding_model}"           

        output_file = (
            f"{output_dir}/output_delta={delta}_gamma={gamma}_k={k}_seed={watermark_seed}"
            f"_watermark={method}_dataset={dataset}{suffix}"
        )
        context["k"] = k
    elif method in {"kgw", "kgw_llr", "unigram"}:
        delta = param["delta"]
        gamma = param["gamma"]
        output_file = (
            f"{output_dir}/output_seed={watermark_seed}_delta={delta}_gamma={gamma}"
            f"_watermark={method}_dataset={dataset}"
        )
    elif method == "distilled":
        delta = param["delta"]
        gamma = param["gamma"]
        output_file = (
            f"{output_dir}/output_seed={watermark_seed}_delta={delta}_gamma={gamma}"
            f"_watermark={method}_dataset={dataset}"
        )
    else:
        raise ValueError(f"Unsupported watermark type {method}.")

    output_file = f"{output_file}.json"

    return output_file, context


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a YAML mapping/object.")
    return loaded


def _build_jobs(cfg: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    method = str(cfg["method"]).strip()

    common = cfg["common"]
    raw_configs = cfg["configs"]
    configs: List[Dict[str, Any]] = [dict(item) for item in raw_configs]

    datasets = common["datasets"]
    watermark_seeds = common.get("watermark_seeds")

    if watermark_seeds is None:
        raise ValueError(f"common.watermark_seeds is required for method={method!r}.")

    jobs: List[Tuple[int, Dict[str, Any]]] = []
    for idx, (dataset, config, seed) in enumerate(
        itertools.product(datasets, configs, watermark_seeds)
    ):
        job = {
            "dataset": dataset,
            **dict(config),
            "watermark_seed": int(seed),
        }
        if method == "distilled":
            job["model_name"] = resolve_distilled_model(job["model_name"], job["watermark_seed"])
        jobs.append((idx, job))

    return jobs


def _run_job_common(args_and_locks: Tuple[Tuple[Any, Dict[str, Any]], Dict[Any, Any], Dict[str, Any]]) -> str:
    (gpu, param), gpu_locks, runtime = args_and_locks

    method = runtime["method"]
    paraphrase = bool(runtime["paraphrase"])
    eval_ppl = bool(runtime["eval_ppl"])
    generation_seed = int(runtime["generation_seed"])
    base_output_dir = runtime["base_output_dir"]
    default_num_samples = int(runtime["num_samples"])

    model_suffix = param["model_name"].split("/")[-1]
    output_dir = f"{base_output_dir}/{model_suffix}"
    log_dir = f"{output_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    num_samples = default_num_samples
    lock = gpu_locks[gpu]
    with lock:
        env = os.environ.copy()
        if isinstance(gpu, tuple):
            gpu = ",".join(map(str, gpu))

        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_CACHE_DISABLE"] = "1"

        try:
            output_file, _ = _build_output_file(
                output_dir=output_dir,
                method=method,
                param=param,
            )
            env["OUTPUT_FILE"] = output_file

            run_summaries: List[str] = []

            generate_kwargs: Dict[str, Any] = {
                **param,
                "num_samples": num_samples,
                "output_file": output_file,
                "watermark": method,
                "generation_seed": generation_seed,
            }
            generate_cmd = [
                sys.executable,
                "-m",
                "scripts.generate_samples",
                *_kwargs_to_cli_args(generate_kwargs),
            ]
            print(f"Executing command: {' '.join(generate_cmd)}")
            rc, stdout, stderr = _run_command(generate_cmd, env)
            generate_log = os.path.join(log_dir, f"generate_{timestamp}.log")
            with open(generate_log, "w", encoding="utf-8") as handle:
                handle.write(stdout)
                handle.write(stderr)
            if rc != 0:
                return (
                    f"Error on GPU {gpu} during generate_samples. "
                    f"See log: {generate_log}\nSTDOUT:\n{stdout.strip()}\nSTDERR:\n{stderr.strip()}"
                )
            run_summaries.append(f"generate_samples: ok ({generate_log})")

            if paraphrase:
                paraphrase_cmd = [
                    sys.executable,
                    "scripts/paraphrase_llm.py",
                    "--output_file",
                    output_file,
                    "--num_beams",
                    "3",
                ]
                print(f"Executing command: {' '.join(paraphrase_cmd)}")
                rc, stdout, stderr = _run_command(paraphrase_cmd, env)
                paraphrase_log = os.path.join(log_dir, f"paraphrase_llm_{timestamp}.log")
                with open(paraphrase_log, "w", encoding="utf-8") as handle:
                    handle.write(stdout)
                    handle.write(stderr)
                if rc != 0:
                    return (
                        f"Error on GPU {gpu} during paraphrase_llm. "
                        f"See log: {paraphrase_log}\nSTDOUT:\n{stdout.strip()}\nSTDERR:\n{stderr.strip()}"
                    )
                run_summaries.append(f"paraphrase_llm: ok ({paraphrase_log})")

            eval_cmd = [
                sys.executable,
                "-m",
                "scripts.test_watermarking_v1",
                "--output_file",
                output_file,
                "--log_dir",
                log_dir,
            ]
            print(f"Executing command: {' '.join(eval_cmd)}")
            rc, stdout, stderr = _run_command(eval_cmd, env)
            eval_log = os.path.join(log_dir, f"tw_{timestamp}.log")
            with open(eval_log, "w", encoding="utf-8") as handle:
                handle.write(stdout)
                handle.write(stderr)
            if rc != 0:
                return (
                    f"Error on GPU {gpu} during test_watermarking_v1. "
                    f"See log: {eval_log}\nSTDOUT:\n{stdout.strip()}\nSTDERR:\n{stderr.strip()}"
                )
            run_summaries.append(f"test_watermarking_v1: ok ({eval_log})")

            if eval_ppl:
                eval_ppl_cmd = [
                    sys.executable,
                    "scripts/evaluate_ppl.py",
                    "--batch_size",
                    "16",
                    "--output_file",
                    output_file,
                ]
                print(f"Executing command: {' '.join(eval_ppl_cmd)}")
                rc, stdout, stderr = _run_command(eval_ppl_cmd, env)
                if rc != 0:
                    return (
                        f"Error on GPU {gpu} during evaluate_ppl."
                        f"\nSTDOUT:\n{stdout.strip()}\nSTDERR:\n{stderr.strip()}"
                    )
                run_summaries.append("evaluate_ppl: ok")

            summary = ", ".join(run_summaries)
            return f"Success on GPU {gpu}: output={output_file}; {summary}"
        except Exception as exc:
            return f"Exception on GPU {gpu}: {exc}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run watermark experiments from a method-specific YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to method YAML config file."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print generated jobs and commands without running subprocesses.",
    )
    parser.add_argument(
        "--base_output_dir",
        required=True,
        help="Base output directory for experiment results.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=42,
        help="Generation seed passed to scripts.generate_samples.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to generate per job.",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="Run paraphrase_llm step.",
    )
    parser.add_argument(
        "--eval_ppl",
        action="store_true",
        help="Run evaluate_ppl step.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config

    raw_cfg = _load_yaml(config_path)
    cfg = raw_cfg

    jobs = _build_jobs(cfg)
    common = cfg["common"]
    method = str(cfg["method"])
    gpus = common["gpus"]
    runtime_common: Dict[str, Any] = {
        "method": method,
        "base_output_dir": args.base_output_dir,
        "num_samples": args.num_samples,
        "generation_seed": args.generation_seed,
        "paraphrase": args.paraphrase,
        "eval_ppl": args.eval_ppl,
    }

    wrapped_jobs = [
        ((gpus[i % len(gpus)], param), None, runtime_common)
        for i, (_, param) in enumerate(jobs)
    ]
    max_workers = max(1, min(len(gpus), len(wrapped_jobs)))

    print(f"Loaded method={method} from {config_path}")
    print(f"Generated {len(wrapped_jobs)} jobs")
    print(f"Using max_workers={max_workers} (inferred)")

    if args.dry_run:
        for idx, ((gpu, param), _, _) in enumerate(wrapped_jobs):
            print(f"[dry_run] job#{idx} gpu={gpu} params={param}")
        return 0

    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    gpu_locks = {gpu: manager.Semaphore(1) for gpu in gpus}

    # Replace placeholder lock dict with real shared locks.
    runnable_jobs = [((gpu, param), gpu_locks, runtime) for (gpu, param), _, runtime in wrapped_jobs]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_job_common, job) for job in runnable_jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as exc:
                print(f"Job failed with exception: {exc}")

    print("All jobs completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
