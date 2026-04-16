from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
TRADER_PATH = Path(__file__).resolve().with_name("root_trader_3.py")
BACKTESTER_PATH = ROOT / ".venv" / "Scripts" / "prosperity4btx.exe"
OUTPUT_DIR = ROOT / "output" / "root-3-tuning"
TEMP_ALGO_DIR = OUTPUT_DIR / "generated_algorithms"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    minimum: float | int
    maximum: float | int | None = None


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("ORDER_SIZE", "int", 1),
    ParamSpec("BASE_K", "float", 0.2),
    ParamSpec("INVENTORY_SKEW", "float", 0.0),
    ParamSpec("MAX_AGGRESSION", "int", 0),
    ParamSpec("STEP_SIZE", "int", 1),
)


def log(message: str) -> None:
    print(message, flush=True)


def default_worker_count() -> int:
    cpu_total = os.cpu_count() or 2
    return max(1, cpu_total - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel coarse-to-fine tuning harness for root_trader_3.py using the venv backtester."
    )
    parser.add_argument(
        "--days",
        nargs="+",
        default=["1"],
        help="Backtester day arguments such as 1 or 1-0. Default: 1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Parallel backtester processes to run at once. Default: CPU count minus one.",
    )
    parser.add_argument(
        "--match-trades",
        choices=["all", "worse", "none"],
        default="all",
        help="Backtester trade matching mode.",
    )
    parser.add_argument(
        "--max-combos-per-stage",
        type=int,
        default=0,
        help="Optional cap for quick smoke tests. 0 means no cap.",
    )
    parser.add_argument(
        "--keep-generated",
        action="store_true",
        help="Keep generated temporary algorithm files instead of deleting them after each run.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing stage checkpoint files and rerun all combos from scratch.",
    )
    parser.add_argument(
        "--prune-top-n",
        type=int,
        default=3,
        help="How many promising values per parameter to keep when pruning later-stage regions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for tuning artifacts.",
    )
    return parser.parse_args()


def load_trader_class() -> Any:
    spec = importlib.util.spec_from_file_location("root_trader_3_module", TRADER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trader from {TRADER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trader


def resolve_defaults(trader_class: Any) -> dict[str, Any]:
    max_position = int(getattr(trader_class, "MAX_POSITION"))
    defaults = {
        "ORDER_SIZE": int(getattr(trader_class, "ORDER_SIZE")),
        "BASE_K": float(getattr(trader_class, "BASE_K")),
        "INVENTORY_SKEW": float(getattr(trader_class, "INVENTORY_SKEW")),
        "MAX_AGGRESSION": int(getattr(trader_class, "MAX_AGGRESSION")),
        "STEP_SIZE": int(getattr(trader_class, "STEP_SIZE")),
    }

    resolved_specs: list[ParamSpec] = []
    for spec in PARAM_SPECS:
        if spec.name == "ORDER_SIZE":
            resolved_specs.append(ParamSpec(spec.name, spec.kind, spec.minimum, max_position))
        else:
            resolved_specs.append(spec)
    globals()["PARAM_SPECS"] = tuple(resolved_specs)
    return defaults


def spec_by_name(name: str) -> ParamSpec:
    for spec in PARAM_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def coarse_delta(spec: ParamSpec, current_value: Any) -> float:
    current = abs(float(current_value))
    delta = min(current * 0.25, 0.2)
    if spec.kind == "int":
        return float(max(1, int(math.ceil(delta))))
    return max(delta, 0.01)


def refined_delta(spec: ParamSpec, base_delta: float, divisor: float) -> float:
    candidate = base_delta / divisor
    if spec.kind == "int":
        return float(max(1, int(math.ceil(candidate))))
    return max(candidate, 0.005)


def cast_param_value(spec: ParamSpec, value: float) -> float | int:
    bounded = max(float(spec.minimum), float(value))
    if spec.maximum is not None:
        bounded = min(float(spec.maximum), bounded)

    if spec.kind == "int":
        return int(round(bounded))
    return round(float(bounded), 4)


def generate_values(
    spec: ParamSpec,
    center: Any,
    delta: float,
    offsets: list[int],
) -> list[float | int]:
    values = [cast_param_value(spec, float(center) + offset * delta) for offset in offsets]
    unique_values = sorted(dict.fromkeys(values))
    return unique_values


def cartesian_product(grids: dict[str, list[Any]]) -> list[dict[str, Any]]:
    items = list(grids.items())
    combos: list[dict[str, Any]] = [{}]
    for name, values in items:
        expanded: list[dict[str, Any]] = []
        for combo in combos:
            for value in values:
                expanded.append({**combo, name: value})
        combos = expanded
    return combos


def build_stage_combos(
    center_params: dict[str, Any],
    target_params: list[str],
    delta_map: dict[str, float],
    offsets: list[int],
) -> list[dict[str, Any]]:
    grids: dict[str, list[Any]] = {}
    for param_name in target_params:
        spec = spec_by_name(param_name)
        grids[param_name] = generate_values(spec, center_params[param_name], delta_map[param_name], offsets)

    combos = []
    for partial in cartesian_product(grids):
        combo = dict(center_params)
        combo.update(partial)
        combos.append(combo)

    unique: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    for combo in combos:
        key = tuple(sorted(combo.items()))
        unique[key] = combo
    return list(unique.values())


def patch_trader_source(source_text: str, params: dict[str, Any]) -> str:
    updated = source_text
    for key, value in params.items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).+$", re.MULTILINE)
        replacement = rf"\g<1>{repr(value)}"
        updated, count = pattern.subn(replacement, updated, count=1)
        if count != 1:
            raise RuntimeError(f"Could not patch parameter {key} in {TRADER_PATH}")
    return updated


def format_combo_label(combo_index: int, params: dict[str, Any]) -> str:
    pieces = [f"{name.lower()}-{str(value).replace('.', 'p')}" for name, value in params.items()]
    return f"combo-{combo_index:05d}__" + "__".join(pieces)


def combo_signature(params: dict[str, Any]) -> str:
    ordered = {name: params[name] for name in sorted(params)}
    return json.dumps(ordered, sort_keys=True, separators=(",", ":"))


def parse_total_profit(stdout: str) -> float:
    match = re.search(r"Total profit:\s*([-\d,]+(?:\.\d+)?)", stdout)
    if match is None:
        raise RuntimeError(f"Could not parse Total profit from backtester output:\n{stdout}")
    return float(match.group(1).replace(",", ""))


def build_backtester_command(
    algorithm_path: Path,
    days: list[str],
    match_trades: str,
) -> list[str]:
    return [
        str(BACKTESTER_PATH),
        str(algorithm_path),
        *days,
        "--match-trades",
        match_trades,
        "--no-out",
        "--no-progress",
    ]


def remove_path_with_retries(path: Path, attempts: int = 5, delay_seconds: float = 0.2) -> None:
    for attempt in range(1, attempts + 1):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            return
        except PermissionError:
            if attempt == attempts:
                return
            time.sleep(delay_seconds * attempt)


def run_backtest_for_combo(
    combo_index: int,
    stage_name: str,
    params: dict[str, Any],
    days: list[str],
    match_trades: str,
    output_dir: Path,
    keep_generated: bool,
) -> dict[str, Any]:
    combo_started = time.perf_counter()
    combo_name = format_combo_label(combo_index, params)
    combo_dir = output_dir / "generated_algorithms" / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)
    algorithm_path = combo_dir / "trader.py"
    source_text = TRADER_PATH.read_text(encoding="utf-8")
    algorithm_path.write_text(patch_trader_source(source_text, params), encoding="utf-8")

    env = os.environ.copy()
    existing_python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing_python_path else os.pathsep.join([str(ROOT), existing_python_path])
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    command = build_backtester_command(algorithm_path, days=days, match_trades=match_trades)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        check=False,
    )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    elapsed_seconds = time.perf_counter() - combo_started

    try:
        if completed.returncode != 0:
            raise RuntimeError(stderr or stdout or f"Backtester exited with code {completed.returncode}")
        total_profit = parse_total_profit(stdout)
        status = "ok"
        error_message = ""
    except Exception as exc:
        total_profit = float("-inf")
        status = "error"
        error_message = str(exc)
    finally:
        if not keep_generated and combo_dir.exists():
            remove_path_with_retries(combo_dir)

    product_pnls: dict[str, float] = {}
    for line in stdout.splitlines():
        if ":" not in line or line.startswith("Backtesting ") or line.startswith("Total profit:"):
            continue
        product_name, value_text = line.split(":", 1)
        value_text = value_text.strip().replace(",", "")
        try:
            product_pnls[product_name.strip()] = float(value_text)
        except ValueError:
            continue

    return {
        "stage": stage_name,
        "combo_index": combo_index,
        "combo_signature": combo_signature(params),
        "status": status,
        "total_profit": total_profit,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "days": ",".join(days),
        "match_trades": match_trades,
        "algorithm_path": str(algorithm_path),
        "stdout": stdout,
        "stderr": stderr,
        "error": error_message,
        **{f"pnl_{name}": value for name, value in product_pnls.items()},
        **params,
    }


def compute_param_impacts(results: pd.DataFrame, param_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean_results = results[results["status"] == "ok"].copy()

    for param_name in param_names:
        grouped = (
            clean_results.groupby(param_name, dropna=False)["total_profit"]
            .agg(["mean", "max", "min", "count"])
            .reset_index()
            .sort_values(param_name)
        )
        if grouped.empty:
            rows.append({"parameter": param_name, "impact_score": float("-inf"), "value_count": 0})
            continue

        impact_score = float(grouped["mean"].max() - grouped["mean"].min())
        peak_lift = float(grouped["max"].max() - grouped["max"].min())
        stability_span = float(grouped["mean"].std(ddof=0)) if len(grouped) > 1 else 0.0
        rows.append(
            {
                "parameter": param_name,
                "impact_score": round(impact_score + 0.25 * peak_lift + 0.1 * stability_span, 4),
                "mean_span": round(impact_score, 4),
                "peak_span": round(peak_lift, 4),
                "stability_span": round(stability_span, 4),
                "value_count": int(grouped["count"].sum()),
                "tested_values": json.dumps(grouped[param_name].tolist()),
            }
        )

    impact_frame = pd.DataFrame(rows).sort_values(["impact_score", "parameter"], ascending=[False, True]).reset_index(
        drop=True
    )
    return impact_frame


def promising_value_sets(
    reference_results: pd.DataFrame,
    param_names: list[str],
    top_n: int,
) -> dict[str, set[Any]]:
    clean = reference_results[reference_results["status"] == "ok"].copy()
    allowed: dict[str, set[Any]] = {}

    for param_name in param_names:
        if clean.empty or param_name not in clean.columns:
            continue
        grouped = (
            clean.groupby(param_name, dropna=False)["total_profit"]
            .agg(["mean", "max", "count"])
            .reset_index()
            .sort_values(["mean", "max", "count"], ascending=[False, False, False])
        )
        if grouped.empty:
            continue
        allowed[param_name] = set(grouped[param_name].head(max(1, top_n)).tolist())
    return allowed


def prune_combos_by_allowed_values(
    combos: list[dict[str, Any]],
    allowed_values: dict[str, set[Any]],
    stage_name: str,
) -> list[dict[str, Any]]:
    if not allowed_values:
        return combos

    pruned: list[dict[str, Any]] = []
    for combo in combos:
        keep = True
        for param_name, allowed in allowed_values.items():
            if allowed and combo[param_name] not in allowed:
                keep = False
                break
        if keep:
            pruned.append(combo)

    if pruned:
        log(f"[{stage_name}] Pruned {len(combos) - len(pruned)} combos using promising-value filters")
        return pruned

    log(f"[{stage_name}] Pruning would remove everything, so keeping all {len(combos)} combos")
    return combos


def load_existing_stage_results(stage_path: Path) -> pd.DataFrame:
    if not stage_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(stage_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def stage_checkpoint_path(output_dir: Path, stage_name: str) -> Path:
    return output_dir / f"{stage_name}_results.csv"


def stage_metadata_path(output_dir: Path, stage_name: str) -> Path:
    return output_dir / f"{stage_name}_checkpoint.json"


def write_stage_metadata(
    metadata_path: Path,
    stage_name: str,
    total_combos: int,
    completed_results: pd.DataFrame,
    pending_count: int,
    status: str,
) -> None:
    payload = {
        "stage": stage_name,
        "status": status,
        "total_combos": int(total_combos),
        "completed_combos": int(len(completed_results)),
        "successful_combos": int((completed_results.get("status", pd.Series(dtype="object")) == "ok").sum()),
        "failed_combos": int((completed_results.get("status", pd.Series(dtype="object")) != "ok").sum()),
        "pending_combos": int(pending_count),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_stage(
    stage_name: str,
    combos: list[dict[str, Any]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.max_combos_per_stage > 0:
        combos = combos[: args.max_combos_per_stage]

    if not combos:
        return pd.DataFrame()

    stage_path = stage_checkpoint_path(args.output_dir, stage_name)
    metadata_path = stage_metadata_path(args.output_dir, stage_name)
    combo_records = [
        {
            "combo_index": combo_index,
            "combo_signature": combo_signature(params),
            "params": params,
        }
        for combo_index, params in enumerate(combos, start=1)
    ]

    existing = pd.DataFrame()
    if not args.no_resume:
        existing = load_existing_stage_results(stage_path)
        if not existing.empty and "combo_signature" not in existing.columns:
            existing["combo_signature"] = existing.apply(
                lambda row: combo_signature({spec.name: row[spec.name] for spec in PARAM_SPECS}),
                axis=1,
            )

    completed_signatures = set(existing.get("combo_signature", pd.Series(dtype="object")).dropna().astype(str))
    pending_records = [record for record in combo_records if record["combo_signature"] not in completed_signatures]
    write_stage_metadata(
        metadata_path,
        stage_name=stage_name,
        total_combos=len(combo_records),
        completed_results=existing,
        pending_count=len(pending_records),
        status="running",
    )

    if not pending_records:
        log(f"[{stage_name}] All {len(combo_records)} combos already completed, resuming from checkpoint")
        frame = existing.sort_values(["status", "total_profit"], ascending=[True, False]).reset_index(drop=True)
        write_stage_metadata(
            metadata_path,
            stage_name=stage_name,
            total_combos=len(combo_records),
            completed_results=frame,
            pending_count=0,
            status="completed",
        )
        return frame

    worker_count = max(1, min(args.workers, len(pending_records)))
    log(
        f"[{stage_name}] Starting {len(pending_records)} pending combos with {worker_count} workers "
        f"({len(completed_signatures)} already checkpointed)"
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = [] if existing.empty else existing.to_dict("records")
    best_profit = float(existing["total_profit"].max()) if not existing.empty and "total_profit" in existing.columns else float("-inf")
    completed = len(completed_signatures)

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                run_backtest_for_combo,
                record["combo_index"],
                stage_name,
                record["params"],
                args.days,
                args.match_trades,
                args.output_dir,
                args.keep_generated,
            ): (record["combo_index"], record["params"])
            for record in pending_records
        }

        for future in concurrent.futures.as_completed(future_map):
            combo_index, params = future_map[future]
            completed += 1
            result = future.result()
            rows.append(result)
            frame = pd.DataFrame(rows).drop_duplicates(subset=["combo_signature"], keep="last")
            frame.to_csv(stage_path, index=False)
            write_stage_metadata(
                metadata_path,
                stage_name=stage_name,
                total_combos=len(combo_records),
                completed_results=frame,
                pending_count=len(combo_records) - len(frame),
                status="running",
            )

            if result["status"] == "ok" and result["total_profit"] > best_profit:
                best_profit = result["total_profit"]

            log(
                f"[{stage_name}] {completed}/{len(combo_records)} "
                f"combo={combo_index} status={result['status']} "
                f"profit={result['total_profit']:.2f} "
                f"elapsed={result['elapsed_seconds']:.2f}s "
                f"best={best_profit:.2f}"
            )

            if result["status"] != "ok":
                log(f"[{stage_name}] combo={combo_index} error={result['error']}")
                log(f"[{stage_name}] combo={combo_index} params={params}")

    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(rows).drop_duplicates(subset=["combo_signature"], keep="last")
    frame = frame.sort_values(["status", "total_profit"], ascending=[True, False]).reset_index(drop=True)
    frame.to_csv(stage_path, index=False)
    write_stage_metadata(
        metadata_path,
        stage_name=stage_name,
        total_combos=len(combo_records),
        completed_results=frame,
        pending_count=0,
        status="completed",
    )
    log(f"[{stage_name}] Complete in {elapsed:.2f}s")
    return frame


def best_result(frame: pd.DataFrame) -> dict[str, Any]:
    clean = frame[frame["status"] == "ok"].copy()
    if clean.empty:
        raise RuntimeError("No successful backtests completed in this stage.")
    row = clean.sort_values(["total_profit", "elapsed_seconds"], ascending=[False, True]).iloc[0]
    return row.to_dict()


def stage_summary_rows(stage_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return [
            {
                "stage": stage_name,
                "successful_runs": 0,
                "failed_runs": int((frame["status"] != "ok").sum()),
                "best_total_profit": None,
                "median_total_profit": None,
                "worst_total_profit": None,
            }
        ]

    return [
        {
            "stage": stage_name,
            "successful_runs": int(len(ok)),
            "failed_runs": int((frame["status"] != "ok").sum()),
            "best_total_profit": float(ok["total_profit"].max()),
            "median_total_profit": float(ok["total_profit"].median()),
            "worst_total_profit": float(ok["total_profit"].min()),
        }
    ]


def write_markdown_summary(
    output_path: Path,
    best_overall: dict[str, Any],
    top_impact_params: list[str],
    stage_summaries: pd.DataFrame,
) -> None:
    lines = [
        "# Root 3 Tuning Summary",
        "",
        "## Best Overall",
        "",
        f"- total_profit: `{best_overall['total_profit']}`",
        f"- order_size: `{best_overall['ORDER_SIZE']}`",
        f"- base_k: `{best_overall['BASE_K']}`",
        f"- inventory_skew: `{best_overall['INVENTORY_SKEW']}`",
        f"- max_aggression: `{best_overall['MAX_AGGRESSION']}`",
        f"- step_size: `{best_overall['STEP_SIZE']}`",
        "",
        "## Highest Impact Parameters",
        "",
    ]
    lines.extend(f"- `{name}`" for name in top_impact_params)
    lines.extend(["", "## Stage Summary", ""])
    for row in stage_summaries.to_dict("records"):
        lines.append(
            f"- `{row['stage']}`: successes=`{row['successful_runs']}`, failures=`{row['failed_runs']}`, "
            f"best=`{row['best_total_profit']}`, median=`{row['median_total_profit']}`, worst=`{row['worst_total_profit']}`"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generated_algorithms").mkdir(parents=True, exist_ok=True)

    if not BACKTESTER_PATH.exists():
        raise FileNotFoundError(f"Backtester not found at {BACKTESTER_PATH}")

    trader_class = load_trader_class()
    defaults = resolve_defaults(trader_class)
    log(f"[setup] Trader defaults={defaults}")
    log(f"[setup] Backtester={BACKTESTER_PATH}")
    log(f"[setup] Days={args.days}")
    log(f"[setup] Resume={'off' if args.no_resume else 'on'}")
    log(f"[setup] Prune top n={args.prune_top_n}")

    base_deltas = {spec.name: coarse_delta(spec, defaults[spec.name]) for spec in PARAM_SPECS}
    log(f"[setup] Coarse deltas={base_deltas}")

    all_param_names = [spec.name for spec in PARAM_SPECS]

    stage1_combos = build_stage_combos(
        center_params=defaults,
        target_params=all_param_names,
        delta_map=base_deltas,
        offsets=[-2, -1, 0, 1, 2],
    )
    stage1 = run_stage("stage1_coarse", stage1_combos, args)
    stage1_path = stage_checkpoint_path(args.output_dir, "stage1_coarse")
    stage1.to_csv(stage1_path, index=False)
    log(f"[stage1_coarse] Wrote {stage1_path}")

    coarse_impacts = compute_param_impacts(stage1, all_param_names)
    coarse_impacts_path = args.output_dir / "stage1_impact_ranking.csv"
    coarse_impacts.to_csv(coarse_impacts_path, index=False)
    log(f"[stage1_coarse] Wrote {coarse_impacts_path}")

    top_impact_params = coarse_impacts["parameter"].head(3).tolist()
    remaining_params = [name for name in all_param_names if name not in top_impact_params]
    stage1_best = best_result(stage1)
    stage2_deltas = {
        name: refined_delta(spec_by_name(name), base_deltas[name], divisor=2.0) for name in top_impact_params
    }

    log(f"[stage1_coarse] Top impact params={top_impact_params}")
    log(f"[stage1_coarse] Remaining params={remaining_params}")

    stage2_combos = build_stage_combos(
        center_params={name: stage1_best[name] for name in all_param_names},
        target_params=top_impact_params,
        delta_map=stage2_deltas,
        offsets=[-3, -2, -1, 0, 1, 2, 3],
    )
    stage2_allowed = promising_value_sets(stage1, top_impact_params, top_n=args.prune_top_n)
    stage2_combos = prune_combos_by_allowed_values(stage2_combos, stage2_allowed, "stage2_top3_refine")
    stage2 = run_stage("stage2_top3_refine", stage2_combos, args)
    stage2_path = stage_checkpoint_path(args.output_dir, "stage2_top3_refine")
    stage2.to_csv(stage2_path, index=False)
    log(f"[stage2_top3_refine] Wrote {stage2_path}")

    stage2_best = best_result(stage2)
    stage3_deltas = {
        name: refined_delta(spec_by_name(name), base_deltas[name], divisor=1.5) for name in remaining_params
    }
    stage3_combos = build_stage_combos(
        center_params={name: stage2_best[name] for name in all_param_names},
        target_params=remaining_params,
        delta_map=stage3_deltas,
        offsets=[-2, -1, 0, 1, 2],
    )
    stage3_allowed = promising_value_sets(stage1, remaining_params, top_n=args.prune_top_n)
    stage3_combos = prune_combos_by_allowed_values(stage3_combos, stage3_allowed, "stage3_remaining_refine")
    stage3 = run_stage("stage3_remaining_refine", stage3_combos, args)
    stage3_path = stage_checkpoint_path(args.output_dir, "stage3_remaining_refine")
    stage3.to_csv(stage3_path, index=False)
    log(f"[stage3_remaining_refine] Wrote {stage3_path}")

    stage3_best = best_result(stage3)
    stage4_deltas = {
        name: refined_delta(spec_by_name(name), stage2_deltas[name], divisor=2.0) for name in top_impact_params
    }
    stage4_combos = build_stage_combos(
        center_params={name: stage3_best[name] for name in all_param_names},
        target_params=top_impact_params,
        delta_map=stage4_deltas,
        offsets=[-2, -1, 0, 1, 2],
    )
    stage4_allowed = promising_value_sets(stage2, top_impact_params, top_n=args.prune_top_n)
    stage4_combos = prune_combos_by_allowed_values(stage4_combos, stage4_allowed, "stage4_top3_fine")
    stage4 = run_stage("stage4_top3_fine", stage4_combos, args)
    stage4_path = stage_checkpoint_path(args.output_dir, "stage4_top3_fine")
    stage4.to_csv(stage4_path, index=False)
    log(f"[stage4_top3_fine] Wrote {stage4_path}")

    combined = pd.concat([stage1, stage2, stage3, stage4], ignore_index=True)
    combined = combined.sort_values(["status", "total_profit"], ascending=[True, False]).reset_index(drop=True)
    combined_path = args.output_dir / "combined_results.csv"
    combined.to_csv(combined_path, index=False)

    stage_summaries = pd.DataFrame(
        stage_summary_rows("stage1_coarse", stage1)
        + stage_summary_rows("stage2_top3_refine", stage2)
        + stage_summary_rows("stage3_remaining_refine", stage3)
        + stage_summary_rows("stage4_top3_fine", stage4)
    )
    stage_summary_path = args.output_dir / "stage_summary.csv"
    stage_summaries.to_csv(stage_summary_path, index=False)

    best_overall = best_result(combined)
    best_params = {name: best_overall[name] for name in all_param_names}
    best_params_path = args.output_dir / "best_params.json"
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    write_markdown_summary(
        args.output_dir / "summary.md",
        best_overall=best_overall,
        top_impact_params=top_impact_params,
        stage_summaries=stage_summaries,
    )

    log(f"[done] Wrote {combined_path}")
    log(f"[done] Wrote {stage_summary_path}")
    log(f"[done] Wrote {best_params_path}")
    log("[done] Best parameter set:")
    log(json.dumps(best_params, indent=2))

    if not args.keep_generated and (args.output_dir / "generated_algorithms").exists():
        try:
            shutil.rmtree(args.output_dir / "generated_algorithms")
        except OSError:
            pass


if __name__ == "__main__":
    main()
