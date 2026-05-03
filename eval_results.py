import pandas as pd
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


def evaluate(
    predictions: list[str],
    gold_path: str = None,
    task: str = None,
    mode: str = None,
):
    lst_pred = []
    for f in predictions:
        lst_pred.append(f)

    test = PyEvALLEvaluation()

    # Select only the entries from gold that correspond to a prediction
    if task is None:
        if "2_1" in predictions[0]:
            task = "2_1"
        elif "2_2" in predictions[0]:
            task = "2_2"
        elif "2_3" in predictions[0]:
            task = "2_3"
        else:
            raise ValueError(
                "Could not infer task from prediction file name. Please specify --task explicitly."
            )
    if mode is None:
        mode = "hard" if "hard" in predictions[0] else "soft"

    if gold_path is None:
        gold_path = (
            f"data/evaluation/golds/EXIST2025_training_task{task}_gold_{mode}.json"
        )

    print("===> Starting evaluation with the following parameters:")
    print(f"Predictions: {predictions}")
    print(f"Gold labels: {gold_path}")
    print(f"Task: {task}")
    print(f"Mode: {mode}\n")

    gold_entries = pd.read_json(gold_path)
    pred_entries = pd.read_json(
        predictions[0]
    )  # HACK: Assumes all prediction files have the same IDs, so we can just read one of them to get the list of IDs
    valid_gold = gold_entries[gold_entries["id"].isin(pred_entries["id"])]
    valid_gold["id"] = valid_gold["id"].astype(str)
    gold_subset_path = gold_path.replace(".json", "_subset.json")
    valid_gold.to_json(gold_subset_path, orient="records", lines=False, indent=2)

    params = dict()
    params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME

    # Define hierarchies based on task
    if task == "2_2":
        params[PyEvALLUtils.PARAM_HIERARCHY] = {
            "YES": ["DIRECT", "JUDGEMENTAL"],
            "NO": [],
        }
    elif task == "2_3":
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
        if task == "2_3":
            metrics = ["ICMSoft", "ICMSoftNorm"]
        else:
            metrics = ["ICMSoft", "ICMSoftNorm", "CrossEntropy"]

    report = test.evaluate_lst(lst_pred, gold_subset_path, metrics, **params)
    report.print_report()
    return


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
        "--predictions_files",
        nargs="+",
        help="Paths to the JSON files containing predictions to evaluate.",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        default=None,
        help="Path to the gold labels file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["2_1", "2_2", "2_3"],
        default=None,
        help="Task identifier.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hard", "soft"],
        default=None,
        help="Evaluation mode: 'hard' for strict metrics, 'soft' for metrics that consider prediction confidence.",
    )
    args = parser.parse_args()

    evaluate(args.predictions_files, args.gold_path, args.task, args.mode)
