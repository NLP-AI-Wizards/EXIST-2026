from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

def evaluate(
    predictions_path: str,
    gold_path: str = "data/evaluation/golds/EXIST2025_training_task2_1_gold_hard.json",
    task: str = "2.1",
    mode: str = "hard",
):
    test = PyEvALLEvaluation()
    params = dict()

    if mode == "hard":
        # HARD eval
        if task == "2.1":
            params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED
            metrics = ["ICM", "ICMNorm", "FMeasure"]
        elif task == "2.2":
            metrics = ["ICM", "ICMNorm", "FMeasure"]
            TASK2_2_HIERARCHY = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
            params[PyEvALLUtils.PARAM_HIERARCHY] = TASK2_2_HIERARCHY
        elif task == "2.3":
            metrics = ["ICM", "ICMNorm", "FMeasure"]
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
        report.print_report()

    else:
        # Soft eval
        if task == "2.1":
            params[PyEvALLUtils.PARAM_REPORT] = (
                PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED
            )
            metrics = ["ICMSoft", "ICMSoftNorm", "CrossEntropy"]
        elif task == "2.2":
            metrics = ["ICMSoft", "ICMSoftNorm", "CrossEntropy"]
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
        report.print_report()

    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate predictions against gold labels.")
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
        help="Task identifier (default: 2.1).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hard", "soft"],
        required=True,
        help="Evaluation mode (default: hard).",
    )
    args = parser.parse_args()
    evaluate(args.predictions_path, args.gold_path, args.task, args.mode)