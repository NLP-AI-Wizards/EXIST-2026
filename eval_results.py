import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


def eval_task(root, file, predictions_dir):
    """Helper function to evaluate a single task file for parallel execution."""
    if not (file.endswith(".json") and "task2_" in file):
        return None

    file_path = os.path.join(root, file)

    # Infer task and mode from filename
    task = None
    if "task2_1" in file:
        task = "2.1"
    elif "task2_2" in file:
        task = "2.2"
    elif "task2_3" in file:
        task = "2.3"
    else:
        return None

    mode = "hard" if "hard" in file.lower() else "soft"

    # Extract model name
    relative_path = os.path.relpath(root, predictions_dir)
    model_name = relative_path if relative_path != "." else "root"

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
            metrics["file"] = file
            metrics["task"] = task
            metrics["mode"] = mode
            return metrics
    except Exception as e:
        print(f"Error evaluating {file}: {e}")
    return None


def eval_all(
    predictions_dir: str = "outputs", output_csv: str = "evaluation_results.csv"
):
    """
    Scans a directory for JSON prediction files and evaluates them all in parallel.
    Generates a CSV report of the results.
    """
    # Collect all candidate files first
    tasks_to_run = []
    for root, _, files in os.walk(predictions_dir):
        for file in files:
            if file.endswith(".json") and "task2_" in file:
                tasks_to_run.append((root, file))

    if not tasks_to_run:
        print("No valid prediction files found for evaluation.")
        return

    print(f"Starting parallel evaluation of {len(tasks_to_run)} files...")

    # Run evaluation in parallel using joblib and tqdm
    results = Parallel(n_jobs=4)(
        delayed(eval_task)(root, file, predictions_dir)
        for root, file in tqdm(tasks_to_run, desc="Evaluating", unit="file")
    )

    # Filter out None results
    results_list = [r for r in results if r is not None]

    if results_list:
        df = pd.DataFrame(results_list)

        # Define categorical order for tasks and modes
        df["task"] = pd.Categorical(
            df["task"], categories=["2.1", "2.2", "2.3"], ordered=True
        )
        df["mode"] = pd.Categorical(
            df["mode"], categories=["soft", "hard"], ordered=True
        )

        # Sort by model, then task, then mode
        df = df.sort_values(by=["model", "task", "mode"])

        # Split and save into two files
        for mode in ["soft", "hard"]:
            mode_df = df[df["mode"] == mode].copy()
            if not mode_df.empty:
                # Remove columns that are all NaN for this mode
                mode_df = mode_df.dropna(axis=1, how="all")

                # Reorder columns to put metadata first
                meta_cols = ["model", "file", "task", "mode"]
                other_cols = [c for c in mode_df.columns if c not in meta_cols]
                mode_df = mode_df[meta_cols + other_cols]

                mode_output = output_csv.replace(".csv", f"_{mode}.csv")
                mode_df.to_csv(mode_output, index=False)
                print(f"\nResults for {mode} mode saved to {mode_output}")
                print(mode_df.to_string())
    else:
        print("No metrics were successfully computed.")


def evaluate(
    predictions_path: str,
    gold_path: str,
    task: str,
    verbose: bool = True,
):
    test = PyEvALLEvaluation()
    params = dict()

    # Determine mode from path
    mode = "hard" if "hard" in predictions_path.lower() else "soft"

    if mode == "hard":
        params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED
        metrics = ["ICM", "ICMNorm", "FMeasure"]
        if task == "2.2":
            TASK2_2_HIERARCHY = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
            params[PyEvALLUtils.PARAM_HIERARCHY] = TASK2_2_HIERARCHY
        elif task == "2.3":
            TASK2_3_HIERARCHY = {
                "YES": [
                    "IDEOLOGICAL-INEQUALITY",
                    "STEREOTYPING-DOMINANCE",
                    "OBJECTIFICATION",
                    "SEXUAL-VIOLENCE",
                    "MISOGYNY-NON-SEXUAL-VIOLENCE",
                ],
                "NO": [],
            }
            params[PyEvALLUtils.PARAM_HIERARCHY] = TASK2_3_HIERARCHY
        report = test.evaluate(predictions_path, gold_path, metrics, **params)

    else:
        params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED
        metrics = ["ICMSoft", "ICMSoftNorm", "CrossEntropy"]
        if task == "2.2":
            TASK2_2_HIERARCHY = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
            params[PyEvALLUtils.PARAM_HIERARCHY] = TASK2_2_HIERARCHY
        elif task == "2.3":
            metrics = ["ICMSoft", "ICMSoftNorm"]
            TASK2_3_HIERARCHY = {
                "YES": [
                    "IDEOLOGICAL-INEQUALITY",
                    "STEREOTYPING-DOMINANCE",
                    "OBJECTIFICATION",
                    "SEXUAL-VIOLENCE",
                    "MISOGYNY-NON-SEXUAL-VIOLENCE",
                ],
                "NO": [],
            }
            params[PyEvALLUtils.PARAM_HIERARCHY] = TASK2_3_HIERARCHY
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
    args = parser.parse_args()

    if args.all:
        eval_all(args.dir)
    elif args.predictions_path and args.gold_path and args.task:
        evaluate(args.predictions_path, args.gold_path, args.task, verbose=True)
    else:
        parser.print_help()
