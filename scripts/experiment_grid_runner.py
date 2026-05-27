import argparse
import csv
import sys
import time
import traceback
import json
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterable, List

from spectral_graph_metric_pavement_cells.core.runtime.logging_config import setup_logging


def ensure_repo_on_path(repo_root: Path):
    rp = str(repo_root.resolve())
    if rp not in sys.path:
        sys.path.insert(0, rp)


def safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"Failed to import {module_name}: {e}")
        return None


def write_master_row(master_csv: Path, row: Dict[str, Any], fieldnames: Iterable[str]):
    write_header = not master_csv.exists() or master_csv.stat().st_size == 0
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(master_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def collect_summaries_and_append(run_dir: Path, combo: Dict[str, Any], master_csv: Path, master_fields: List[str]):
    found = list(run_dir.rglob("summaries_master.csv"))
    if not found:
        print("  -> Warning: no summaries_master.csv found in", run_dir)
        return
    for sm in found:
        try:
            import csv as _csv
            with open(sm, newline="", encoding="utf-8") as rf:
                r = _csv.DictReader(rf)
                for row in r:
                    cfg_name = (row.get("config") or "").lower()
                    classifier = None
                    if "logreg" in cfg_name or "logistic" in cfg_name:
                        classifier = "logistic_regression"
                    elif "rf" in cfg_name or "random" in cfg_name:
                        classifier = "random_forest"

                    acc_mean = row.get("acc_mean") or row.get("acc_fold_mean") or ""
                    acc_std = row.get("acc_std") or row.get("acc_fold_std") or ""
                    prec_macro = row.get("prec_macro_mean") or row.get("prec_fold_mean") or ""
                    rec_macro = row.get("rec_macro_mean") or row.get("rec_fold_mean") or ""
                    f1_macro = row.get("f1_macro_mean") or row.get("f1_fold_mean") or ""

                    master_row = {
                        "timestamp_utc": datetime.now().isoformat() + "Z",
                        "dataset": combo["dataset_name"],
                        "analysis_feature_key": combo["analysis_feature_key"],
                        "weight_func": combo["weight_func"],
                        "node_count": combo["node_count"],
                        "scaler_name": combo["scaler_name"],
                        "normalize_by": combo["normalize_by"] or "",
                        "classifier": classifier,
                        "acc_mean": acc_mean,
                        "acc_std": acc_std,
                        "prec_macro_mean": prec_macro,
                        "rec_macro_mean": rec_macro,
                        "f1_macro_mean": f1_macro,
                        "n_samples": row.get("n_samples") or "",
                        "n_features": row.get("dim") or row.get("n_features") or "",
                        "n_classes": row.get("n_classes") or "",
                        "cv_folds": row.get("cv_folds") or "",
                        "run_dir": str(run_dir),
                    }

                    # 1. Capture Per-Fold Metrics (Acc, F1, Recall)
                    for k in range(10):  # Support up to 10 folds
                        key_acc = f"acc_fold_{k}"
                        key_prec = f"prec_fold_{k}"
                        key_rec = f"rec_fold_{k}"
                        key_f1 = f"f1_fold_{k}"


                        if key_acc in row: master_row[key_acc] = row[key_acc]
                        if key_prec in row: master_row[key_prec] = row[key_prec]
                        if key_rec in row: master_row[key_rec] = row[key_rec]
                        if key_f1 in row:  master_row[key_f1] = row[key_f1]

                    # 2. Capture Per-Class Metrics
                    for k, v in row.items():
                        if k.startswith(("f1_", "rec_", "prec_", "feat_imp_")) and k not in master_row:
                            master_row[k] = v

                    write_master_row(master_csv, master_row, master_fields)
        except Exception as e:
            print(f"  -> Failed reading summary {sm}: {e}")


def run_single_combo(
        combo: Dict[str, Any],
        config_module_name: str,
        build_experiment,
        run_pipeline,
        RESULTS_DIR: Path,
        experiment_timestamp: str,
):
    # Import the configuration module inside the worker process
    cfg_mod = importlib.import_module(config_module_name)

    ts = experiment_timestamp
    norm_label = combo["normalize_by"] if combo["normalize_by"] is not None else "None"
    run_dirname = (
        f"{combo['dataset_name']}_{combo['analysis_feature_key']}_{combo['weight_func']}_"
        f"{combo['node_count']}_{combo['scaler_name']}_{norm_label}_{ts}"
    )

    run_dir = Path(RESULTS_DIR) / "grid_search" / run_dirname
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        comp_pipeline, analysis_pipeline = build_experiment(
            dataset_name=combo["dataset_name"],
            analysis_feature_key=combo["analysis_feature_key"],
            weight_func=combo["weight_func"],
            node_count=combo["node_count"],
            scaler_name=combo["scaler_name"],
            normalize_by=combo["normalize_by"],
        )
    except Exception as e:
        print(f"  -> Error building experiment: {e}\n{traceback.format_exc()}")
        return None

    result_list = {}
    for cfg in comp_pipeline:
        try:
            save_path = None
            if cfg.get("save", False):
                save_path = str(run_dir / cfg["name"])
                Path(save_path).mkdir(parents=True, exist_ok=True)
            res = run_pipeline(
                reader=cfg["reader"],
                graph=cfg["graph"],
                distance=cfg["distance"],
                data_path=cfg["data_path"],
                save_path=save_path,
                analysis_feature_key=cfg["analysis_feature_key"],
                weight_func=cfg["weight_func_name"],
                plot_contour=cfg.get("plot_contour", False),
                filter_strategy=cfg.get("filter_strategy", None),
            )
            result_list[cfg["name"]] = res
        except Exception as e:
            print(f"    -> Error running pipeline {cfg['name']}: {e}\n{traceback.format_exc()}")

    for idx, ap in enumerate(getattr(cfg_mod, "analysis_pipeline", analysis_pipeline) or analysis_pipeline):
        cp = analysis_pipeline[idx] if idx < len(analysis_pipeline) else ap
        try:
            cp.set_results(result_list)
        except Exception:
            if hasattr(cp, "analysis_strategy") and hasattr(cp.analysis_strategy, "set_results"):
                cp.analysis_strategy.set_results(result_list)

        strat = getattr(cp, "analysis_strategy", cp)
        cfg_name = getattr(cp, "config_name", f"{combo['dataset_name']}_{combo['analysis_feature_key']}_{ts}")
        try:
            if hasattr(strat, "set_save_info"):
                strat.set_save_info(str(run_dir), cfg_name)
        except Exception:
            pass

        try:
            print(f"  - analysis: {type(strat).__name__}")
            if hasattr(cp, "run_analysis") and callable(cp.run_analysis):
                res = cp.run_analysis()
            elif hasattr(strat, "analyze") and callable(strat.analyze):
                res = strat.analyze(result_list)
            else:
                raise RuntimeError("No runnable analysis method found for pipeline")

            with open(run_dir / f"analysis_{idx}_{type(strat).__name__}.json", "w", encoding="utf-8") as jf:
                json.dump(res, jf, indent=2, default=str)
        except Exception as e:
            print(f"    -> Error during analysis {type(strat).__name__}: {e}\n{traceback.format_exc()}")

    try:
        import spectral_graph_metric_pavement_cells.core.runtime.metadata as meta_mod
        if hasattr(meta_mod, "save_pipeline_metadata"):
            meta_mod.save_pipeline_metadata(
                computation_pipeline=comp_pipeline,
                analysis_pipeline=analysis_pipeline,
                save_path=str(run_dir),
                config_name=f"{combo['dataset_name']}_{combo['analysis_feature_key']}_{ts}",
            )
    except Exception:
        pass

    return run_dir


def run_grid(repo_root: Path, config_module_name: str, outdir: Path, max_workers: int = None):
    ensure_repo_on_path(repo_root)

    cfg_mod = safe_import(config_module_name)
    if cfg_mod is None:
        raise RuntimeError(f"Could not import experiment config module '{config_module_name}'")

    required = [
        "DATASETS",
        "ANALYSIS_FEATURE_KEYS",
        "WEIGHT_FUNCS",
        "NODE_COUNTS",
        "SCALERS",
        "NORMALIZE_BY_OPTIONS",
        "build_experiment",
    ]
    missing = [n for n in required if not hasattr(cfg_mod, n)]
    if missing:
        raise RuntimeError(f"Experiment config module missing required names: {missing}")

    DATASETS = getattr(cfg_mod, "DATASETS")
    ANALYSIS_FEATURE_KEYS = getattr(cfg_mod, "ANALYSIS_FEATURE_KEYS")
    WEIGHT_FUNCS = getattr(cfg_mod, "WEIGHT_FUNCS")
    NODE_COUNTS = getattr(cfg_mod, "NODE_COUNTS")
    SCALERS = getattr(cfg_mod, "SCALERS")
    NORMALIZE_BY_OPTIONS = getattr(cfg_mod, "NORMALIZE_BY_OPTIONS")
    build_experiment = getattr(cfg_mod, "build_experiment")
    RESULTS_DIR = getattr(cfg_mod, "RESULTS_DIR", outdir)

    pipelines_mod = safe_import("spectral_graph_metric_pavement_cells.core.pipelines.pipeline_runner")
    if pipelines_mod is None:
        raise RuntimeError("Could not import spectral_graph_metric_pavement_cells.core.pipelines.pipeline_runner")
    run_pipeline = getattr(pipelines_mod, "run_pipeline", None)
    if run_pipeline is None:
        raise RuntimeError("run_pipeline not found in pipeline_runner module")

    master_csv = Path(RESULTS_DIR) / "grid_search" / "master_summary.csv"

    # --- Define Master Fields including Per-Fold metrics ---
    master_fields = [
        "timestamp_utc",
        "dataset",
        "analysis_feature_key",
        "weight_func",
        "node_count",
        "scaler_name",
        "normalize_by",
        "classifier",
        "acc_mean",
        "acc_std",
        "prec_macro_mean",
        "rec_macro_mean",
        "f1_macro_mean",
        "n_samples",
        "n_features",
        "n_classes",
        "cv_folds",
        "run_dir",
    ]
    master_fields.extend([f"acc_fold_{i}" for i in range(10)])
    master_fields.extend([f"prec_fold_{i}" for i in range(10)])
    master_fields.extend([f"rec_fold_{i}" for i in range(10)])
    master_fields.extend([f"f1_fold_{i}" for i in range(10)])
    master_fields.extend([f"feat_imp_{i}" for i in range(140)])

    combos = []
    for dataset_name in DATASETS.keys():
        for afk in ANALYSIS_FEATURE_KEYS:
            if afk == "eigvals_L" or afk == "eigvals_Q" or afk == "eigvals_Lsym" or afk == "eigvals_Qsym" :
                current_weight_funcs = [None]
            else:
                current_weight_funcs = WEIGHT_FUNCS
            for weight in current_weight_funcs:
                if weight == "distance_norm" :
                    normalize_iter = NORMALIZE_BY_OPTIONS
                else:
                    normalize_iter = [None]
                for normalize_by in normalize_iter:
                    for node_count in NODE_COUNTS:
                        for scaler_name in SCALERS:
                            combos.append(
                                {
                                    "dataset_name": dataset_name,
                                    "analysis_feature_key": afk,
                                    "weight_func": weight,
                                    "node_count": node_count,
                                    "scaler_name": scaler_name,
                                    "normalize_by": normalize_by,
                                }
                            )

    print(f"Total combinations: {len(combos)}")

    max_workers = max_workers or 6
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {
            executor.submit(
                run_single_combo,
                combo,
                config_module_name,
                build_experiment,
                run_pipeline,
                RESULTS_DIR,
                experiment_timestamp,
            ): combo
            for combo in combos
        }

        for i, future in enumerate(as_completed(future_to_combo), 1):
            combo = future_to_combo[future]
            try:
                run_dir = future.result()
                if run_dir is not None:
                    collect_summaries_and_append(run_dir, combo, master_csv, master_fields)
                    print(f"[{i}/{len(combos)}] Completed: {run_dir}")
                else:
                    print(f"[{i}/{len(combos)}] Failed combo: {combo}")
            except Exception as e:
                print(f"[{i}/{len(combos)}] Exception for combo {combo}: {e}")

    print(f"\nAll experiments done. Master summary location: {master_csv}")
    return master_csv


def main():
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument("--config-module", default="scripts.configs.grid_search_config", help="Experiment config module")
    parser.add_argument("--outdir", default="data/results/grid_search", help="Root output directory")
    parser.add_argument("--max-workers", type=int, default=None, help="Number of parallel workers (processes)")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    outdir = (repo_root / args.outdir).resolve()
    print("Repo root:", repo_root)
    print("Experiment config module:", args.config_module)

    start_time = time.time()

    master = run_grid(repo_root, args.config_module, outdir, max_workers=args.max_workers)
    print("Master CSV:", master)

    end_time = time.time()
    elapsed_time = end_time - start_time

    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"\nTotal execution time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")


if __name__ == "__main__":
    main()