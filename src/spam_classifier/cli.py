"""Command line: ``python -m spam_classifier train | evaluate | predict``."""
import argparse
import datetime
import sys
from pathlib import Path

from .data import DEFAULT_DATA_PATH, REPO_ROOT, load_dataset, split_dataset
from .evaluation import compute_metrics
from .features import FEATURE_NAMES
from .models import DEFAULT_ALGORITHM, DEFAULT_FEATURES, MODEL_NAMES, build_pipeline
from .predict import DEFAULT_MODEL_PATH, load_pipeline, predict_messages, save_pipeline


def _print_metrics(metrics, n_test, n_spam):
    print(f"test set: {n_test} messages ({n_spam} spam)")
    print(f"accuracy {metrics['accuracy']:.6f}")
    print(f"precision (spam) {metrics['precision']:.6f}")
    print(f"recall (spam) {metrics['recall']:.6f}")
    print(f"f1 (spam) {metrics['f1']:.6f}")
    cm = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    print(f"confusion matrix [[TN, FP], [FN, TP]]: {cm}")


def _cmd_train(args):
    df = load_dataset(args.data)
    x_train, x_test, y_train, y_test = split_dataset(df)
    pipe = build_pipeline(
        args.algorithm, args.features, normalize=args.normalize, class_weight=args.class_weight
    )
    pipe.fit(x_train.tolist(), y_train)
    save_pipeline(pipe, args.output)
    print(f"trained {args.algorithm} on {args.features} features ({len(x_train)} training messages)")
    print(f"saved {args.output}")
    metrics = compute_metrics(y_test, pipe.predict(x_test.tolist()))
    _print_metrics(metrics, len(x_test), int(y_test.sum()))
    return 0


def _cmd_evaluate(args):
    if args.compare:
        from .report import run_report

        run_report(
            data_path=args.data,
            out_dir=args.out_dir,
            figures_dir=args.figures_dir,
            run_date=args.date,
        )
        return 0
    pipe = load_pipeline(args.model_path)
    _, x_test, _, y_test = split_dataset(load_dataset(args.data))
    metrics = compute_metrics(y_test, pipe.predict(x_test.tolist()))
    _print_metrics(metrics, len(x_test), int(y_test.sum()))
    return 0


def _cmd_predict(args):
    messages = list(args.messages)
    if args.file:
        messages += [line for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not messages:
        print("error: give at least one message, or --file", file=sys.stderr)
        return 2
    pipe = load_pipeline(args.model_path)
    for message, pred in zip(messages, predict_messages(pipe, messages, threshold=args.threshold), strict=True):
        shown = message if len(message) <= 70 else message[:67] + "..."
        print(f"{pred.label}\t{pred.score:.4f}\t{shown}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="spam_classifier", description="SMS spam classifier.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train on the notebook's 70% split and save the model")
    train.add_argument("--algorithm", choices=MODEL_NAMES, default=DEFAULT_ALGORITHM)
    train.add_argument("--features", choices=FEATURE_NAMES, default=DEFAULT_FEATURES)
    train.add_argument("--normalize", action="store_true", help="replace URLs, prices and phone numbers by tokens")
    train.add_argument("--class-weight", choices=["balanced"], default=None, help="logistic regression and SVM only")
    train.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    train.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    train.set_defaults(func=_cmd_train)

    evaluate = sub.add_parser("evaluate", help="evaluate a saved model, or run the full comparison with --compare")
    evaluate.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    evaluate.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    evaluate.add_argument("--compare", action="store_true", help="compare all models and write the run logs and figures")
    evaluate.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "runs"), help="where --compare writes run logs")
    evaluate.add_argument("--figures-dir", default=str(REPO_ROOT / "docs" / "figures"))
    evaluate.add_argument("--date", default=datetime.date.today().isoformat(), help="run date used in file names and labels")
    evaluate.set_defaults(func=_cmd_evaluate)

    predict = sub.add_parser("predict", help="classify messages with a saved model")
    predict.add_argument("messages", nargs="*", help="message texts")
    predict.add_argument("--file", help="a text file with one message per line")
    predict.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    predict.add_argument("--threshold", type=float, default=None, help="flag as spam when score >= threshold")
    predict.set_defaults(func=_cmd_predict)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
