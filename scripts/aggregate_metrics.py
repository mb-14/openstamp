#!/usr/bin/env python3
"""Aggregate PPL, AUROC, TPR@FPR, mean_seq_rep_3 across seeds for JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Union


SEED_RE = re.compile(r"^(?P<prefix>.*)_seed=(?P<seed>[^_]+)(?P<suffix>.*)\.json$")


def _parse_json(path: Path) -> Dict[str, Any]:
    def _parse_constant(val: str) -> float:
        if val == "NaN":
            return float("nan")
        if val == "Infinity":
            return float("inf")
        if val == "-Infinity":
            return float("-inf")
        raise ValueError(f"Unexpected constant: {val}")

    with path.open("r", encoding="utf-8") as f:
        return json.loads(f.read(), parse_constant=_parse_constant)


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def _extract_group_id(filename: str) -> str:
    match = SEED_RE.match(filename)
    if not match:
        return filename.replace(".json", "")
    return f"{match.group('prefix')}{match.group('suffix')}"


def _extract_params(data: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    config = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
    alignment_method = config.get("align_method")
    formatted_config = _format_config(config)
    embedding_model = config.get("embedding_model")
    
    if embedding_model:
        if formatted_config:
            formatted_config = f"{formatted_config},embedding_model={embedding_model}"
        else:
            formatted_config = f"embedding_model={embedding_model}"
    if alignment_method:
        if formatted_config:
            formatted_config = f"{formatted_config},align_method={alignment_method}"
        else:
            formatted_config = f"align_method={alignment_method}"
    return {
        "config": formatted_config,
        "dataset": data.get("dataset_name"),
        "method": data.get("watermark"),
    }


def _format_config(config: Dict[str, Any]) -> Optional[str]:
    mb_keys = [
        ("k", config.get("n_clusters")),
        ("gamma", config.get("gamma")),
        ("delta", config.get("delta")),
    ]
    gauss_keys = [
        ("sigma", config.get("sigma")),
        ("target_param_name", config.get("target_param_name")),
    ]

    if any(value is not None for _, value in mb_keys):
        items = [f"{key}={value}" for key, value in mb_keys if value is not None]
        return ",".join(items) if items else None
    if any(value is not None for _, value in gauss_keys):
        items = [f"{key}={value}" for key, value in gauss_keys if value is not None]
        return ",".join(items) if items else None
    return None


def _metrics_prefix(metrics_key: str) -> str:
    if metrics_key == "metrics":
        return "gen"
    if "lex20" in metrics_key:
        return "dipper20"
    if "lex60" in metrics_key:
        return "dipper60"
    if "llm_paraphrase" in metrics_key:
        return "llm_paraphrase"
    return metrics_key


def _collect_metrics(data: Dict[str, Any], metrics_keys: List[str]) -> Dict[str, Optional[float]]:
    ppl = data.get("ppl", {}) if isinstance(data.get("ppl"), dict) else {}
    result: Dict[str, Optional[float]] = {
        "ppl_mean": _safe_number(ppl.get("mean")),
        "mean_seq_rep_3": _safe_number(data.get("mean_seq_rep_3")),
    }

    for metrics_key in metrics_keys:
        metrics = data.get(metrics_key, {}) if isinstance(data.get(metrics_key), dict) else {}
        prefix = _metrics_prefix(metrics_key)
        result[f"{prefix}_auroc"] = _safe_number(metrics.get("auroc"))

        for key, value in metrics.items():
            if isinstance(key, str) and key.startswith("tpr_") and key.endswith("_fpr"):
                result[f"{prefix}_{key}"] = _safe_number(value)

    return result


def _aggregate_rows(
    rows: List[Dict[str, Optional[float]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    keys: Iterable[str] = set().union(*(row.keys() for row in rows))
    aggregated: Dict[str, Dict[str, Optional[float]]] = {}
    for key in keys:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if not values:
            aggregated[key] = {"mean": None, "std": None, "stderr": None}
            continue

        metric_mean = mean(values)
        metric_std = stdev(values) if len(values) > 1 else 0.0
        metric_stderr = metric_std / math.sqrt(len(values)) if len(values) > 0 else None
        aggregated[key] = {
            "mean": metric_mean,
            "std": metric_std,
            "stderr": metric_stderr,
        }
    return aggregated


def _format_aggregated_value(
    metric: Dict[str, Optional[float]], error_bar: str
) -> Optional[Union[float, str]]:
    metric_mean = metric.get("mean")
    metric_std = metric.get("std")
    metric_stderr = metric.get("stderr")

    if metric_mean is None:
        return None

    def _fmt(num: Optional[float]) -> str:
        if num is None:
            return ""
        return f"{num:.3f}"

    if error_bar == "none":
        return _fmt(metric_mean)
    if error_bar == "std":
        return f"{_fmt(metric_mean)} +- {_fmt(metric_std)}"
    if error_bar == "stderr":
        return f"{_fmt(metric_mean)} +- {_fmt(metric_stderr)}"
    if error_bar == "both":
        return f"{_fmt(metric_mean)} +- {_fmt(metric_std)} (stderr={_fmt(metric_stderr)})"

    return _fmt(metric_mean)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics across seeds.")
    parser.add_argument(
        "--input-dir",
        default="output/unigram",
        help=(
            "Directory inside a parent that contains multiple output directories. "
            "All subdirectories in the parent will be scanned for JSON files."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Path to write aggregated CSV. If omitted, uses "
            "<input-dir>/aggregated_metrics.csv."
        ),
    )
    parser.add_argument(
        "--metrics-keys",
        default=(
            "metrics,metrics_dipper_text_lex20_order0,metrics_dipper_text_lex60_order0,"
            "metrics_llm_paraphrase"
        ),
        help="Comma-separated top-level keys that contain AUROC/TPR metrics.",
    )
    parser.add_argument(
        "--error-bar",
        default="std",
        choices=["none", "std", "stderr", "both"],
        help=(
            "How to include uncertainty across seeds in averaged metric columns: "
            "none, std, stderr, or both. Format is 'mean +- error'."
        ),
    )
    args = parser.parse_args()

    metrics_keys = [key.strip() for key in args.metrics_keys.split(",") if key.strip()]

    input_dir = Path(args.input_dir)
    if args.output_csv is None:
        args.output_csv = str(input_dir / "aggregated_metrics.csv")
    json_files_set = set()
    # Scan all subdirectories under input_dir for JSON files
    if input_dir.exists():
        for model_dir in input_dir.iterdir():
            if not model_dir.is_dir():
                continue
            json_files_set.update(p for p in model_dir.glob("*.json") if p.is_file())
    json_files = sorted(json_files_set)

    grouped: Dict[str, List[Dict[str, Optional[float]]]] = {}
    grouped_params: Dict[str, Dict[str, Optional[Any]]] = {}
    group_model: Dict[str, str] = {}
    for path in json_files:
        data = _parse_json(path)
        # Model is the immediate parent directory name
        model_name = path.parent.name
        # Make group_id unique by including model_name
        group_id = f"{model_name}/{_extract_group_id(path.name)}"
        metrics = _collect_metrics(data, metrics_keys)
        grouped.setdefault(group_id, []).append(metrics)
        if group_id not in grouped_params:
            grouped_params[group_id] = _extract_params(data)
        group_model[group_id] = model_name

    # Build output rows
    all_metric_keys = set()
    for rows in grouped.values():
        for row in rows:
            all_metric_keys.update(row.keys())

    # Stable column ordering
    prefixes = ["gen", "dipper20", "dipper60", "llm_paraphrase"]
    metric_cols: List[str] = []
    tpr_suffixes = ["1_fpr", "0.01_fpr", "0.1_fpr"]
    for prefix in prefixes:
        auroc_col = f"{prefix}_auroc"
        if auroc_col in all_metric_keys:
            metric_cols.append(auroc_col)
        for tpr_suffix in tpr_suffixes:
            tpr_col = f"{prefix}_tpr_{tpr_suffix}"
            if tpr_col in all_metric_keys:
                metric_cols.append(tpr_col)

    base_numeric_cols = ["ppl_mean", "mean_seq_rep_3", *metric_cols]

    columns = [
        "model",
        "dataset",
        "method",
        "config",
        *base_numeric_cols,
    ]

    output_rows: List[Dict[str, Any]] = []
    for group_id, rows in sorted(grouped.items()):
        aggregated = _aggregate_rows(rows)
        out_row: Dict[str, Any] = {}
        out_row["model"] = group_model.get(group_id, "")
        out_row.update(grouped_params.get(group_id, {}))

        for col in base_numeric_cols:
            out_row[col] = _format_aggregated_value(aggregated.get(col, {}), args.error_bar)

        output_rows.append(out_row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        def _metric_group(col: str) -> str:
            if col.endswith("_auroc"):
                return col[: -len("_auroc")]
            if "_tpr_" in col and col.endswith("_fpr"):
                return col.split("_tpr_")[0]
            return ""

        def _metric_label(col: str) -> str:
            if col.endswith("_auroc"):
                return "auroc"
            if "_tpr_" in col and col.endswith("_fpr"):
                return f"tpr_{col.split('_tpr_')[1]}"
            return col

        base_cols = [
            "model",
            "dataset",
            "method",
            "config",
        ]
        metric_cols = [c for c in columns if c not in base_cols]
        header1 = [*base_cols, *(_metric_group(c) if c not in {"ppl_mean", "mean_seq_rep_3"} else ("ppl" if c == "ppl_mean" else "mean_seq_rep_3") for c in metric_cols)]
        header2 = [*( [""] * len(base_cols)), *(_metric_label(c) if c not in {"ppl_mean", "mean_seq_rep_3"} else "mean" for c in metric_cols)]
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(header2)
        for row in output_rows:
            writer.writerow([row.get(col) for col in columns])


if __name__ == "__main__":
    main()
