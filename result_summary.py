"""Atomic model-level mean summaries for skeleton experiments.

Raw per-sample JSON files remain the source of truth.  The runners call this
module at the end of a model run to materialise a small, plot-ready summary at
``results/mean_results/<result_family>/.../mean_summary.json``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import glob
import json
import math
import os
import tempfile
from typing import Any, Iterable


SUMMARY_FORMAT_VERSION = 1
_ORI_ACV_SUFFIX = " (ori_acv)"

_PER_NEURON_DISPLAY_TO_COND = {
    "PerNeuron.top(gate)": "top",
    "PerNeuron.rnd(gate)": "random",
    "PerNeuron.mid(gate)": "middle",
    "PerNeuron.bot(gate)": "bottom",
    "PerNeuron.top_eff(gate)": "top_eff",
    "PerNeuron.rnd_eff(gate)": "random_eff",
    "PerNeuron.bot_eff(gate)": "bottom_eff",
    "PerNeuron.top(nogate)": "top_nogate",
    "PerNeuron.rnd(nogate)": "random_nogate",
    "PerNeuron.bot(nogate)": "bottom_nogate",
    "PerNeuron.top_abla(gate)": "top_abla_gate",
    "PerNeuron.top_abla(nogate)": "top_abla_nogate",
    "Original": "original",
}

_STATIC_DISPLAY_TO_COND = {
    "Ske.pre(gate)": "ske_preserve_gate",
    "Ske.pre(gate,posEff)": "ske_preserve_gate_pos_eff",
    "Ske.pre(no-gate)": "ske_preserve_no_gate",
    "Ske.abl(no-gate)": "ske_ablation_no_gate",
    "Ske.abl(gate)": "ske_ablation_gate",
    "Rnd.pre(gate)": "rnd_preserve_gate",
    "Rnd.pre(no-gate)": "rnd_preserve_no_gate",
    "Rnd.abl(no-gate)": "rnd_ablation_no_gate",
    "Rnd.abl(gate)": "rnd_ablation_gate",
    "Anti-align.pre": "anti_align",
    "Ske.pre(gate,wEff)": "ske_preserve_gate_weight_eff",
    "Ske.pre(gate,Eff)": "ske_preserve_gate_eff",
    "Ske.pre(gate,sqrtEff)": "ske_preserve_gate_sqrteff",
    "Ske.pre(softgate)": "ske_preserve_gate_softgate",
    "Ske.pre(gate,wDelta)": "ske_preserve_gate_weight_delta",
    "Ske.pre(gate,sqrtDelta)": "ske_preserve_gate_sqrtdelta",
    "Ske.pre(effDom)": "ske_preserve_gate_eff_dominate",
    "Ske.pre(effDom,thr)": "ske_preserve_gate_eff_dominate_threh",
    "Eff.dRank.pre(gate)": "eff_delta_rank_pre",
    "Eff.dRank.abl(gate)": "eff_delta_rank_abl",
    "Eff.dwRank.pre(gate)": "eff_delta_weight_rank_pre",
    "Eff.dwRank.abl(gate)": "eff_delta_weight_rank_abl",
    "Original": "original",
}


def _normalise_dynamic_label(label: str) -> str:
    if label.endswith(_ORI_ACV_SUFFIX):
        return label[: -len(_ORI_ACV_SUFFIX)]
    return label


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _percentile(values: list[float], percentile: float) -> float:
    """NumPy-compatible linear percentile for the simple rank use case."""
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _robust_rank_metrics(ranks: Iterable[Any]) -> dict[str, float]:
    clean = [float(v) for v in ranks if _is_number(v)]
    if not clean:
        return {}
    ordered = sorted(clean)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    cutoff = _percentile(ordered, 99.0)
    trimmed = [value for value in ordered if value <= cutoff]
    return {
        "median_rank": median,
        "top1_acc": sum(value == 1 for value in ordered) / len(ordered),
        "trimmed_mean_rank": sum(trimmed) / len(trimmed),
    }


def _mean_metric_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Average label/metric rows, ignoring missing values.

    Model-level rows are deliberately averaged from per-target rows.  This
    preserves the plotting code's existing equal-target weighting rather than
    allowing targets with more samples to dominate.
    """
    accumulated: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        for label, metrics in row.items():
            for metric, value in metrics.items():
                if _is_number(value):
                    accumulated[label][metric].append(float(value))
    return {
        label: {
            metric: sum(values) / len(values)
            for metric, values in metric_values.items()
            if values
        }
        for label, metric_values in accumulated.items()
    }


def _atomic_json_dump(path: str, data: dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".mean_summary.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _summary_path(output_dir: str, model_name: str, scope: str) -> str:
    output_dir = os.path.normpath(output_dir)
    result_family = os.path.basename(output_dir)
    results_root = os.path.dirname(output_dir)
    safe_model = model_name.replace("/", "_")
    return os.path.join(
        results_root,
        "mean_results",
        result_family,
        safe_model,
        scope,
        "mean_summary.json",
    )


def _read_json_files(raw_dir: str) -> tuple[list[tuple[str, dict]], list[str]]:
    documents = []
    errors = []
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if not data.get("summary_type"):
                documents.append((path, data))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{os.path.basename(path)}: {exc}")
    return documents, errors


def write_dynamic_model_mean(
    *,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    expected_items: list[dict],
    k_percents: list[float],
    start_layer: int,
    end_layer: int,
    rely_ori_acv: bool,
    conditions: Iterable[str],
    prompt_format_version: int,
    result_format_version: int,
    sample_indices: list[int] | None = None,
) -> tuple[str, bool]:
    """Write one dynamic model/dataset summary and return ``(path, complete)``."""
    safe_model = model_name.replace("/", "_")
    raw_dir = os.path.join(output_dir, safe_model)
    if dataset_name != "default":
        raw_dir = os.path.join(raw_dir, dataset_name)
    documents, read_errors = _read_json_files(raw_dir)
    condition_tuple = tuple(conditions)
    k_values = {float(value) for value in k_percents}
    indices = sample_indices or list(range(1, len(expected_items) + 1))
    expected = {(int(index), float(k)) for index in indices for k in k_values}
    expected_items_by_index = {
        int(index): item for index, item in zip(indices, expected_items)
    }

    selected: dict[tuple[int, float], tuple[float, dict]] = {}
    for path, data in documents:
        actual_dataset = data.get("dataset", "default")
        key = (data.get("sample_idx"), data.get("k_percent"))
        try:
            normalised_key = (int(key[0]), float(key[1]))
        except (TypeError, ValueError):
            continue
        expected_item = expected_items_by_index.get(normalised_key[0])
        actual_target = data.get("target_word", data.get("target_token"))
        actual_prompt = data.get("in_seq", data.get("prompt"))
        if (
            data.get("model_name") != model_name
            or actual_dataset != dataset_name
            or data.get("prompt_format_version") != prompt_format_version
            or data.get("result_format_version") != result_format_version
            or data.get("start_layer") != start_layer
            or data.get("end_layer") != end_layer
            or bool(data.get("rely_ori_acv")) != bool(rely_ori_acv)
            or tuple(data.get("conditions", [])) != condition_tuple
            or normalised_key not in expected
            or expected_item is None
            or actual_target != expected_item.get("target_token", expected_item.get("target_word"))
            or actual_prompt != expected_item.get("prompt", expected_item.get("in_seq"))
        ):
            continue
        modified = os.path.getmtime(path)
        if modified >= selected.get(normalised_key, (-1.0, {}))[0]:
            selected[normalised_key] = (modified, data)

    # target -> k -> label -> metric -> samples
    accumulated = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    target_sample_ids = defaultdict(lambda: defaultdict(set))
    for (_sample_index, k_percent), (_modified, data) in selected.items():
        target = data.get("target_word", data.get("target_token", "unknown")).strip()
        target_sample_ids[target][k_percent].add(_sample_index)
        for summary_row in data.get("summary", []):
            label = _normalise_dynamic_label(summary_row.get("label", "unknown"))
            for metric in (
                "rank", "logit", "prob", "family_rank", "family_logit", "family_prob"
            ):
                value = summary_row.get(metric)
                if _is_number(value):
                    accumulated[target][k_percent][label][metric].append(float(value))

    by_target: dict[str, dict[str, Any]] = {}
    for target, k_dict in accumulated.items():
        target_results = []
        for k_percent in sorted(k_dict):
            metrics = {}
            for label, metric_values in k_dict[k_percent].items():
                label_metrics = {
                    metric: sum(values) / len(values)
                    for metric, values in metric_values.items()
                    if values
                }
                label_metrics.update(_robust_rank_metrics(metric_values.get("rank", [])))
                family_robust = _robust_rank_metrics(metric_values.get("family_rank", []))
                if "top1_acc" in family_robust:
                    label_metrics["family_top1_acc"] = family_robust["top1_acc"]
                metrics[label] = label_metrics
            target_results.append({
                "k_percent": k_percent,
                "n_samples": len(target_sample_ids[target][k_percent]),
                "metrics": metrics,
            })
        by_target[target] = {
            "n_samples": max(
                (entry["n_samples"] for entry in target_results), default=0
            ),
            "results": target_results,
        }

    mean_results = []
    for k_percent in sorted(k_values):
        target_entries = [
            next(
                (entry for entry in target_data["results"]
                 if float(entry["k_percent"]) == k_percent),
                None,
            )
            for target_data in by_target.values()
        ]
        target_entries = [entry for entry in target_entries if entry is not None]
        mean_results.append({
            "k_percent": k_percent,
            "n_targets": len(target_entries),
            "n_samples": sum(entry["n_samples"] for entry in target_entries),
            "metrics": _mean_metric_rows(entry["metrics"] for entry in target_entries),
        })

    missing = sorted(expected - set(selected))
    complete = not missing and bool(expected)
    # rely_ori_acv=True must not share a path with the False variant (same
    # model/dataset) — otherwise whichever run finishes last silently
    # clobbers the other's mean_summary.json.  Matches the seq_type suffix
    # convention load_dynamic_means_for_plot already uses at read time.
    scope = f"{dataset_name}__ori_acv" if rely_ori_acv else dataset_name
    path = _summary_path(output_dir, model_name, scope)
    payload = {
        "summary_type": "dynamic_skeleton_model_mean",
        "summary_format_version": SUMMARY_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset": dataset_name,
        "effector_mode": "raw_dot",
        "configuration": {
            "prompt_format_version": prompt_format_version,
            "result_format_version": result_format_version,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "rely_ori_acv": bool(rely_ori_acv),
            "conditions": list(condition_tuple),
            "k_percents": sorted(k_values),
        },
        "source": {
            "raw_directory": raw_dir,
            "expected_result_count": len(expected),
            "aggregated_result_count": len(selected),
            "complete": complete,
            "missing_result_count": len(missing),
            "missing_results_preview": [
                {"sample_idx": index, "k_percent": k} for index, k in missing[:20]
            ],
            "read_errors": read_errors[:20],
        },
        "by_target": by_target,
        "mean_results": mean_results,
    }
    _atomic_json_dump(path, payload)
    return path, complete


def write_per_neuron_model_mean(
    *,
    output_dir: str,
    model_name: str,
    seq_type: str,
    targets: Iterable[str],
    k_per_neuron_values: Iterable[int],
    start_layer: int,
    end_layer: int,
    conditions: Iterable[str],
    prompt_format_version: int,
    result_format_version: int,
) -> tuple[str, bool]:
    """Write one static per-neuron model/sequence summary."""
    target_values = [target.strip().lstrip() for target in targets]
    k_values = {int(value) for value in k_per_neuron_values}
    expected = {(target, k) for target in target_values for k in k_values}
    safe_model = model_name.replace("/", "_")
    raw_dir = os.path.join(output_dir, safe_model, seq_type)
    documents, read_errors = _read_json_files(raw_dir)
    condition_tuple = tuple(conditions)

    selected: dict[tuple[str, int], tuple[float, dict]] = {}
    for path, data in documents:
        try:
            key = (data.get("target_word", "unknown").strip().lstrip(), int(data["k_per_neuron"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            data.get("model_name") != model_name
            or data.get("prompt_format_version") != prompt_format_version
            or data.get("result_format_version") != result_format_version
            or data.get("start_layer") != start_layer
            or data.get("end_layer") != end_layer
            or tuple(data.get("conditions", [])) != condition_tuple
            or key[1] not in k_values
        ):
            continue
        modified = os.path.getmtime(path)
        if modified >= selected.get(key, (-1.0, {}))[0]:
            selected[key] = (modified, data)

    by_target: dict[str, dict[str, Any]] = {}
    # Include every compatible target already present in the model directory.
    # This makes repeated single-target CLI calls build one cumulative model
    # summary, matching the old plotting loader's view of the raw directory.
    selected_targets = sorted({target for target, _kpn in selected})
    for target in selected_targets:
        target_results = []
        for kpn in sorted(k_values):
            selected_entry = selected.get((target, kpn))
            if selected_entry is None:
                continue
            data = selected_entry[1]
            per_prompt_ranks: dict[str, list] = defaultdict(list)
            per_prompt_family_ranks: dict[str, list] = defaultdict(list)
            for prompt_result in data.get("per_prompt", []):
                for condition, values in prompt_result.items():
                    if not isinstance(values, dict):
                        continue
                    if _is_number(values.get("rank")):
                        per_prompt_ranks[condition].append(values["rank"])
                    if _is_number(values.get("family_rank")):
                        per_prompt_family_ranks[condition].append(values["family_rank"])

            metrics = {}
            for summary_row in data.get("summary", []):
                label = summary_row.get("label", "unknown")
                condition = _PER_NEURON_DISPLAY_TO_COND.get(label, label.lower())
                label_metrics = {
                    key: float(value)
                    for key, value in summary_row.items()
                    if key != "label" and _is_number(value)
                }
                label_metrics.update(_robust_rank_metrics(per_prompt_ranks.get(condition, [])))
                family_robust = _robust_rank_metrics(
                    per_prompt_family_ranks.get(condition, [])
                )
                if "top1_acc" in family_robust:
                    label_metrics["family_top1_acc"] = family_robust["top1_acc"]
                metrics[label] = label_metrics
            target_results.append({
                "k_per_neuron": kpn,
                "n_samples": int(data.get("n_samples", len(data.get("per_prompt", [])))),
                "metrics": metrics,
            })
        by_target[target] = {
            "n_samples": max(
                (entry["n_samples"] for entry in target_results), default=0
            ),
            "results": target_results,
        }

    mean_results = []
    for kpn in sorted(k_values):
        target_entries = [
            next(
                (entry for entry in target_data["results"]
                 if int(entry["k_per_neuron"]) == kpn),
                None,
            )
            for target_data in by_target.values()
        ]
        target_entries = [entry for entry in target_entries if entry is not None]
        mean_results.append({
            "k_per_neuron": kpn,
            "n_targets": len(target_entries),
            "n_samples_per_target": (
                min((entry["n_samples"] for entry in target_entries), default=0)
            ),
            "metrics": _mean_metric_rows(entry["metrics"] for entry in target_entries),
        })

    missing = sorted(expected - set(selected))
    complete = not missing and bool(expected)
    path = _summary_path(output_dir, model_name, seq_type)
    payload = {
        "summary_type": "static_per_neuron_model_mean",
        "summary_format_version": SUMMARY_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "seq_type": seq_type,
        "effector_mode": "raw_dot",
        "configuration": {
            "prompt_format_version": prompt_format_version,
            "result_format_version": result_format_version,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "conditions": list(condition_tuple),
            "k_per_neuron_values": sorted(k_values),
        },
        "source": {
            "raw_directory": raw_dir,
            "expected_result_count": len(expected),
            "aggregated_result_count": len(selected),
            "complete": complete,
            "missing_result_count": len(missing),
            "missing_results_preview": [
                {"target": target, "k_per_neuron": kpn}
                for target, kpn in missing[:20]
            ],
            "read_errors": read_errors[:20],
        },
        "by_target": by_target,
        "mean_results": mean_results,
    }
    _atomic_json_dump(path, payload)
    return path, complete


def write_static_model_mean(
    *,
    output_dir: str,
    model_name: str,
    start_layer: int,
    end_layer: int,
    baselines: Iterable[str],
    prompt_format_version: int,
    result_format_version: int,
) -> tuple[str, bool]:
    """Rebuild the cumulative global-static summary for one model.

    The global-static API runs one target/k point per invocation, so it has no
    explicit sweep boundary.  After each point is saved we summarise every
    compatible point currently present for that model.
    """
    safe_model = model_name.replace("/", "_")
    raw_dir = os.path.join(output_dir, safe_model)
    documents, read_errors = _read_json_files(raw_dir)
    baseline_tuple = tuple(sorted(baselines))
    selected: dict[tuple[str, float], tuple[float, dict]] = {}
    for path, data in documents:
        k_percent = data.get("k_percent")
        if not _is_number(k_percent):
            continue
        target = data.get("target_word", "unknown").strip().lstrip()
        key = (target, float(k_percent))
        if (
            data.get("model_name") != model_name
            or data.get("prompt_format_version") != prompt_format_version
            or data.get("result_format_version") != result_format_version
            or data.get("start_layer") != start_layer
            or data.get("end_layer") != end_layer
            or tuple(sorted(data.get("baselines", []))) != baseline_tuple
        ):
            continue
        modified = os.path.getmtime(path)
        if modified >= selected.get(key, (-1.0, {}))[0]:
            selected[key] = (modified, data)

    by_target: dict[str, dict[str, Any]] = {}
    targets = sorted({target for target, _k in selected})
    k_values = sorted({k for _target, k in selected})
    for target in targets:
        target_results = []
        for k_percent in k_values:
            selected_entry = selected.get((target, k_percent))
            if selected_entry is None:
                continue
            data = selected_entry[1]
            per_prompt_ranks: dict[str, list] = defaultdict(list)
            per_prompt_family_ranks: dict[str, list] = defaultdict(list)
            for prompt_result in data.get("per_prompt", []):
                for condition, values in prompt_result.items():
                    if not isinstance(values, dict):
                        continue
                    if _is_number(values.get("rank")):
                        per_prompt_ranks[condition].append(values["rank"])
                    if _is_number(values.get("family_rank")):
                        per_prompt_family_ranks[condition].append(values["family_rank"])

            metrics = {}
            for summary_row in data.get("summary", []):
                label = summary_row.get("label", "unknown")
                condition = _STATIC_DISPLAY_TO_COND.get(label, label.lower())
                label_metrics = {
                    key: float(value)
                    for key, value in summary_row.items()
                    if key != "label" and _is_number(value)
                }
                label_metrics.update(_robust_rank_metrics(per_prompt_ranks.get(condition, [])))
                family_robust = _robust_rank_metrics(
                    per_prompt_family_ranks.get(condition, [])
                )
                if "top1_acc" in family_robust:
                    label_metrics["family_top1_acc"] = family_robust["top1_acc"]
                metrics[label] = label_metrics
            target_results.append({
                "k_percent": k_percent,
                "n_samples": int(data.get("n_samples", len(data.get("per_prompt", [])))),
                "metrics": metrics,
            })
        by_target[target] = {
            "n_samples": max(
                (entry["n_samples"] for entry in target_results), default=0
            ),
            "results": target_results,
        }

    mean_results = []
    for k_percent in k_values:
        target_entries = [
            next(
                (entry for entry in target_data["results"]
                 if float(entry["k_percent"]) == k_percent),
                None,
            )
            for target_data in by_target.values()
        ]
        target_entries = [entry for entry in target_entries if entry is not None]
        mean_results.append({
            "k_percent": k_percent,
            "n_targets": len(target_entries),
            "n_samples_per_target": min(
                (entry["n_samples"] for entry in target_entries), default=0
            ),
            "metrics": _mean_metric_rows(entry["metrics"] for entry in target_entries),
        })

    complete = bool(selected)
    path = _summary_path(output_dir, model_name, "default")
    payload = {
        "summary_type": "static_skeleton_model_mean",
        "summary_format_version": SUMMARY_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "effector_mode": "raw_dot",
        "configuration": {
            "prompt_format_version": prompt_format_version,
            "result_format_version": result_format_version,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "baselines": list(baseline_tuple),
        },
        "source": {
            "raw_directory": raw_dir,
            "aggregated_result_count": len(selected),
            "complete": complete,
            "completion_semantics": "all currently available compatible target/k results",
            "read_errors": read_errors[:20],
        },
        "by_target": by_target,
        "mean_results": mean_results,
    }
    _atomic_json_dump(path, payload)
    return path, complete


def load_static_means_for_plot(
    results_dir: str, prompt_format_version: int | None = None
) -> tuple[dict, list[dict]]:
    """Load global-static summaries into plot_dym_stat's legacy shape."""
    result = defaultdict(lambda: defaultdict(dict))
    source_stats = []
    pattern = os.path.join(
        results_dir, "mean_results", "static_ske", "*", "*", "mean_summary.json"
    )
    for path in glob.glob(pattern):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            data.get("summary_type") != "static_skeleton_model_mean"
            or not data.get("source", {}).get("complete")
            or (
                prompt_format_version is not None
                and data.get("configuration", {}).get("prompt_format_version")
                != prompt_format_version
            )
        ):
            continue
        model = data["model_name"]
        effector_mode = data.get("effector_mode", "raw_dot")
        for target, target_data in data.get("by_target", {}).items():
            result[target][model][effector_mode] = [
                (entry["k_percent"], entry["metrics"])
                for entry in target_data.get("results", [])
            ]
        source_stats.append({
            "model_name": model,
            "n_results": data.get("source", {}).get("aggregated_result_count", 0),
        })
    return {
        target: {model: dict(mode_dict) for model, mode_dict in model_dict.items()}
        for target, model_dict in result.items()
    }, source_stats


def load_dynamic_means_for_plot(
    results_dir: str, prompt_format_version: int | None = None
) -> tuple[dict, list[dict]]:
    """Load complete dynamic summaries into plot_dym_stat's legacy shape."""
    result = defaultdict(lambda: defaultdict(dict))
    source_stats = []
    patterns = [
        os.path.join(results_dir, "mean_results", family, "*", "*", "mean_summary.json")
        for family in ("dynamic_ske", "dynamic_factual_ske")
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                data.get("summary_type") != "dynamic_skeleton_model_mean"
                or not data.get("source", {}).get("complete")
                or (
                    prompt_format_version is not None
                    and data.get("configuration", {}).get("prompt_format_version")
                    != prompt_format_version
                )
            ):
                continue
            model = data["model_name"]
            dataset = data.get("dataset", "default")
            seq_type = next(
                (name for name in ("short", "long", "random") if name in dataset),
                dataset,
            )
            if data.get("configuration", {}).get("rely_ori_acv"):
                seq_type = f"{seq_type}__ori_acv"
            for target, target_data in data.get("by_target", {}).items():
                result[target][model][seq_type] = [
                    (entry["k_percent"], entry["metrics"])
                    for entry in target_data.get("results", [])
                ]
            source_stats.append({
                "model_name": model,
                "seq_type": seq_type,
                "n_samples": data.get("source", {}).get("aggregated_result_count", 0)
                // max(len(data.get("configuration", {}).get("k_percents", [])), 1),
            })
    return {
        target: {model: dict(seq_dict) for model, seq_dict in model_dict.items()}
        for target, model_dict in result.items()
    }, source_stats


def load_per_neuron_means_for_plot(
    results_dir: str, prompt_format_version: int | None = None,
    enabled_models: set[str] | None = None,
) -> tuple[dict, list[dict]]:
    """Load complete static per-neuron summaries into the legacy plot shape."""
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    source_stats = []
    pattern = os.path.join(
        results_dir,
        "mean_results",
        "static_ske_per_neuron",
        "*",
        "*",
        "mean_summary.json",
    )
    enabled_dirs = (
        {model.replace("/", "_") for model in enabled_models}
        if enabled_models is not None else None
    )
    for path in glob.glob(pattern):
        model_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if enabled_dirs is not None and model_dir not in enabled_dirs:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            data.get("summary_type") != "static_per_neuron_model_mean"
            or not data.get("source", {}).get("complete")
            or (
                prompt_format_version is not None
                and data.get("configuration", {}).get("prompt_format_version")
                != prompt_format_version
            )
        ):
            continue
        model = data["model_name"]
        seq_type = data.get("seq_type", "default")
        effector_mode = data.get("effector_mode", "raw_dot")
        for target, target_data in data.get("by_target", {}).items():
            result[target][model][seq_type][effector_mode] = [
                (entry["k_per_neuron"], entry["metrics"])
                for entry in target_data.get("results", [])
            ]
        source_stats.append({
            "model_name": model,
            "seq_type": seq_type,
            "targets": list(data.get("by_target", {})),
            "n_samples": min(
                (target.get("n_samples", 0) for target in data.get("by_target", {}).values()),
                default=0,
            ),
        })
    return {
        target: {
            model: {
                seq_type: dict(mode_dict)
                for seq_type, mode_dict in seq_dict.items()
            }
            for model, seq_dict in model_dict.items()
        }
        for target, model_dict in result.items()
    }, source_stats
