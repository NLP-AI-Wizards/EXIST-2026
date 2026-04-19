from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


def evaluate(
    predictions_path: str,
    gold_path: str = "data/evaluation/golds/EXIST2025_training_task2_1_gold_hard.json",
    task: str = "2.1",
    verbose: bool = True,
):
    test = PyEvALLEvaluation()
    params = dict()

    # Infer mode from filename convention
    if "hard" in predictions_path.lower() and "hard" in gold_path.lower():
        mode = "hard"
    elif "soft" in predictions_path.lower() and "soft" in gold_path.lower():
        mode = "soft"
    else:
        raise ValueError(
            "Could not infer evaluation mode from file names. Please ensure both predictions and gold files contain 'hard' or 'soft' in their names."
        )

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate predictions against gold labels."
    )
    parser.add_argument(
        "predictions_path", type=str, help="Path to the predictions file (JSON format)."
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        required=True,
        help="Path to the gold labels file (JSON format).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["2.1", "2.2", "2.3"],
        required=True,
        help="Task identifier.",
    )
    args = parser.parse_args()
    evaluate(args.predictions_path, args.gold_path, args.task, verbose=True)
