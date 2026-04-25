import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


def eval_task(file_path: str, file_name: str, model_name: str):
    """Helper function to evaluate a single task file for parallel execution."""
    if not (file_name.endswith(".json") and "task2_" in file_name):
        return None

    # Infer task and mode from filename
    task = None
    if "task2_1" in file_name:
        task = "2.1"
    elif "task2_2" in file_name:
        task = "2.2"
    elif "task2_3" in file_name:
        task = "2.3"
    else:
        return None

    mode = "hard" if "hard" in file_name.lower() else "soft"

    # Define gold path
    gold_task = task.replace(".", "_")
    gold_path = (
        f"data/evaluation/golds/EXIST2025_training_task{gold_task}_gold_{mode}.json"
    )

    if not os.path.exists(gold_path):
        return None

    try:
        metrics = evaluate(file_path, gold_path=gold_path, task=task, verbose=False)
        if metrics:
            metrics["model"] = model_name
            metrics["file"] = file_name
            metrics["task"] = task
            metrics["mode"] = mode
            return metrics
    except Exception as e:
        print(f"Error evaluating {file_name}: {e}")
    return None


def eval_all(
    predictions_dir: str = "outputs",
    baselines_dir: str = "data/evaluation/baselines",
    include_baselines: bool = True,
    output_csv: str = "evaluation_results.csv",
    overwrite: bool = False,
):
    """
    Scans a directory for JSON prediction files and evaluates them all in parallel.
    Generates a CSV report of the results.
    """
    existing_results = []

    for mode in ["soft", "hard"]:
        mode_csv = output_csv.replace(".csv", f"_{mode}.csv")
        if os.path.exists(mode_csv):
            try:
                existing_results.extend(pd.read_csv(mode_csv).to_dict("records"))
            except Exception as e:
                print(f"Warning: Could not read {mode_csv}: {e}")

    def infer_task(file_name: str):
        if "task2_1" in file_name:
            return "2.1"
        if "task2_2" in file_name:
            return "2.2"
        if "task2_3" in file_name:
            return "2.3"
        return None

    def infer_mode(file_name: str):
        return "hard" if "hard" in file_name.lower() else "soft"

    def make_key(model: str, file_name: str, task: str, mode: str):
        return (str(model), str(file_name), str(task), str(mode))

    existing_by_key = {}
    for row in existing_results:
        row_task = row.get("task")
        row_mode = row.get("mode")
        if row_task is None or row_mode is None:
            continue
        key = make_key(row.get("model", ""), row.get("file", ""), row_task, row_mode)
        existing_by_key[key] = row

    tasks_to_run = []

    # Model outputs under the predictions directory.
    for root, _, files in os.walk(predictions_dir):
        relative_path = os.path.relpath(root, predictions_dir)
        model_name = relative_path if relative_path != "." else "root"

        for file in files:
            if file.endswith(".json") and "task2_" in file:
                task = infer_task(file)
                mode = infer_mode(file)
                if task is None:
                    continue

                key = make_key(model_name, file, task, mode)
                if overwrite or key not in existing_by_key:
                    file_path = os.path.join(root, file)
                    tasks_to_run.append((file_path, file, model_name))

    # Baseline files are included as pseudo-models to ease ranking comparisons.
    if include_baselines and os.path.isdir(baselines_dir):
        for file in os.listdir(baselines_dir):
            if not (file.endswith(".json") and "task2_" in file and "training" in file):
                continue

            task = infer_task(file)
            mode = infer_mode(file)
            if task is None:
                continue

            if "majority" in file.lower():
                model_name = "baseline_majority"
            elif "minority" in file.lower():
                model_name = "baseline_minority"
            else:
                model_name = "baseline"

            key = make_key(model_name, file, task, mode)
            if overwrite or key not in existing_by_key:
                file_path = os.path.join(baselines_dir, file)
                tasks_to_run.append((file_path, file, model_name))

    if not tasks_to_run:
        print("No new prediction files found for evaluation.")
        results_list = list(existing_by_key.values())
    else:
        print(f"Starting parallel evaluation of {len(tasks_to_run)} new files...")
        new_results = Parallel(n_jobs=args.njobs)(
            delayed(eval_task)(file_path, file_name, model_name)
            for file_path, file_name, model_name in tqdm(
                tasks_to_run, desc="Evaluating", unit="file"
            )
        )

        for row in new_results:
            if row is None:
                continue
            key = make_key(row["model"], row["file"], row["task"], row["mode"])
            existing_by_key[key] = row

        results_list = list(existing_by_key.values())

    if not results_list:
        print("No metrics were successfully computed.")
        return

    df = pd.DataFrame(results_list)
    df["task"] = pd.Categorical(
        df["task"].astype(str), categories=["2.1", "2.2", "2.3"], ordered=True
    )
    df["mode"] = pd.Categorical(df["mode"], categories=["soft", "hard"], ordered=True)

    for mode in ["soft", "hard"]:
        mode_df = df[df["mode"] == mode].copy()
        if mode_df.empty:
            continue

        sort_metric = "ICMSoft" if mode == "soft" else "ICM"
        if sort_metric in mode_df.columns:
            # Sort by task, then metric decreasing
            mode_df = mode_df.sort_values(
                by=["task", sort_metric], ascending=[True, False]
            )

        mode_df = mode_df.dropna(axis=1, how="all")

        # Reorder columns to put metadata first
        meta_cols = ["task", "model", "file", "mode"]
        other_cols = [c for c in mode_df.columns if c not in meta_cols]
        mode_df = mode_df[meta_cols + other_cols]

        mode_output = output_csv.replace(".csv", f"_{mode}.csv")
        # Save flat CSV for easy loading later
        mode_df.to_csv(mode_output, index=False)

        print(f"\nResults for {mode} mode saved to {mode_output}")
        # Set a multi-index for cleaner terminal output
        display_df = mode_df.drop(columns=["mode"]).set_index(["task", "model"])
        print(display_df.to_string())


def evaluate(
    predictions_path: str,
    gold_path: str,
    task: str,
    verbose: bool = True,
):
    test = PyEvALLEvaluation()
    mode = "hard" if "hard" in predictions_path.lower() else "soft"

    params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED}

    # Define hierarchies based on task
    if task == "2.2":
        params[PyEvALLUtils.PARAM_HIERARCHY] = {
            "YES": ["DIRECT", "JUDGEMENTAL"],
            "NO": [],
        }
    elif task == "2.3":
        params[PyEvALLUtils.PARAM_HIERARCHY] = {
            "YES": [
                "IDEOLOGICAL-INEQUALITY",
                "STEREOTYPING-DOMINANCE",
                "OBJECTIFICATION",
                "SEXUAL-VIOLENCE",
                "MISOGYNY-NON-SEXUAL-VIOLENCE",
            ],
            "NO": [],
        }

    # Define metrics based on mode and task
    if mode == "hard":
        metrics = ["ICM", "ICMNorm", "FMeasure"]
    else:
        metrics = (
            ["ICMSoft", "ICMSoftNorm"]
            if task == "2.3"
            else ["ICMSoft", "ICMSoftNorm", "CrossEntropy"]
        )

    report = test.evaluate(predictions_path, gold_path, metrics, **params)

    if verbose:
        report.print_report()

    # Extract metrics from the report object safely
    metrics_dict = {}
    try:
        if hasattr(report, "report") and "metrics" in report.report:
            for metric_name, metric_data in report.report["metrics"].items():
                if "results" in metric_data:
                    results = metric_data["results"]
                    if "average_per_test_case" in results:
                        metrics_dict[metric_name] = results["average_per_test_case"]
                    elif "average" in results:
                        metrics_dict[metric_name] = results["average"]
    except Exception as e:
        print(f"Error during metric extraction: {e}")

    return metrics_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate predictions against gold labels."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run evaluation on all files in the outputs directory.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="outputs",
        help="Directory to scan for JSON files when using --all.",
    )
    parser.add_argument(
        "--baselines-dir",
        type=str,
        default="data/evaluation/baselines",
        help="Directory containing baseline JSON predictions for task2.*.",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Do not include baseline files when using --all.",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        help="Path to a single predictions file.",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        help="Path to the gold labels file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["2.1", "2.2", "2.3"],
        help="Task identifier.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing CSV files when using --all.",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=8,
        help="Number of parallel jobs to run when using --all.",
    )
    args = parser.parse_args()

    if args.all:
        eval_all(
            predictions_dir=args.dir,
            baselines_dir=args.baselines_dir,
            include_baselines=not args.no_baselines,
            overwrite=args.overwrite,
        )
    elif args.predictions_path and args.gold_path and args.task:
        evaluate(args.predictions_path, args.gold_path, args.task, verbose=True)
    else:
        parser.print_help()
