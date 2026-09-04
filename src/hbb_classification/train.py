"""CLI for the P1 leakage-controlled pipeline (and the original notebook path)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hbb_classification.config import DEFAULT_CONFIG_PATH, load_config
from hbb_classification.data import load_raw_csv
from hbb_classification.experiment import run_p1_pipeline
from hbb_classification.logutil import setup_logging
from hbb_classification.pipeline import run_notebook_pipeline


def _print_legacy(results: dict) -> None:
    metrics = results["metrics"]
    labels = [
        ("naive_bayes", "Naive Bayes"),
        ("lda", "LDA"),
        ("logistic_regression", "Logistic Regression"),
        ("logistic_regression_pca", "Logistic Regression (PCA)"),
        ("logistic_regression_rf_top8", "Logistic Regression (RF top 8)"),
    ]
    for key, title in labels:
        if key not in metrics:
            continue
        block = metrics[key]
        print(f"\n{title}")
        print(
            f"  accuracy={block['accuracy']:.4f}  "
            f"signal_f1={block['signal']['f1']:.4f}  "
            f"background_f1={block['background']['f1']:.4f}"
        )
    punzi_all = results["punzi_all_features"]
    punzi_pca = results["punzi_pca"]
    print(
        f"\nPunzi (all features): threshold={punzi_all['best_threshold']:.3f}  "
        f"score={punzi_all['best_score']:.6f}"
    )
    print(
        f"Punzi (PCA):          threshold={punzi_pca['best_threshold']:.3f}  "
        f"score={punzi_pca['best_score']:.6f}"
    )


def _fmt_rejection(block: dict) -> str:
    rejection = block.get("background_rejection")
    if rejection is None:
        return "inf" if block.get("reached") else "n/a"
    return f"{rejection:.2f}"


def _print_p1(results: dict) -> None:
    split = results["split"]
    print(
        f"Split train/val/test: {split['n_train']:,} / {split['n_val']:,} / {split['n_test']:,} "
        f"(natural class prior)"
    )
    print(
        f"Test counts: background={split['class_counts']['test']['background']:,}  "
        f"signal={split['class_counts']['test']['signal']:,}"
    )

    physics = results["physics_baseline"]
    print(
        f"\nPhysics baseline: {physics['feature']} ({physics['direction']})  "
        f"val AUC={physics['val']['roc_auc']:.4f}  test AUC={physics['test']['roc_auc']:.4f}"
    )
    print(
        f"  test rejection@50%={_fmt_rejection(physics['test']['rejection_at_50pct_eff'])}  "
        f"@80%={_fmt_rejection(physics['test']['rejection_at_80pct_eff'])}"
    )

    order = [
        ("naive_bayes", "Naive Bayes"),
        ("lda", "LDA"),
        ("logistic_regression", "Logistic Regression"),
        ("logistic_regression_pca", "Logistic Regression (PCA)"),
        ("random_forest", "Random Forest"),
    ]
    for key, title in order:
        if key not in results["models"]:
            continue
        block = results["models"][key]
        test = block["test"]
        punzi = block["punzi_threshold_from_val"]
        print(f"\n{title}")
        print(
            f"  val AUC={block['val']['roc_auc']:.4f}  test AUC={test['roc_auc']:.4f}  "
            f"test AP={test['average_precision']:.4f}"
        )
        print(
            f"  test rejection@50%={_fmt_rejection(test['rejection_at_50pct_eff'])}  "
            f"@80%={_fmt_rejection(test['rejection_at_80pct_eff'])}"
        )
        frozen = test["at_punzi_threshold"]
        print(
            f"  Punzi threshold from val={punzi['threshold']:.3f}; "
            f"test Punzi={frozen['punzi']:.6f}  "
            f"signal F1={frozen['signal_f1']:.4f}  "
            f"accuracy={frozen['accuracy']:.4f}"
        )
        if "n_components" in block:
            print(
                f"  pca_components={block['n_components']}  "
                f"explained_variance={block['explained_variance']:.4f}"
            )

    ablation = results["sentinel"]["ablation"]
    print("\nSentinel ablation (validation AUC, logistic regression)")
    print(f"  raw -1 left in:          {ablation['val_roc_auc_raw_sentinels']:.6f}")
    print(f"  indicator + median impute: {ablation['val_roc_auc_with_sentinel_handling']:.6f}")
    print(f"\nSelected by validation AUC: {results['selected_by_validation_auc']}")


def build_parser(argv: list[str] | None = None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config (default: configs/default.yaml)",
    )
    pre_args, _remaining = pre.parse_known_args(argv)
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train and evaluate Hbb jet classifiers (P1 methodology by default).",
        parents=[pre],
    )
    parser.add_argument("--data", type=Path, default=config.data_path)
    parser.add_argument("--output", type=Path, default=config.metrics_path)
    parser.add_argument("--figure-dir", type=Path, default=config.figure_dir)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        default=not config.check_hash,
        help="Skip the SHA-256 check (still checks size and schema)",
    )
    parser.add_argument(
        "--skip-random-forest",
        action="store_true",
        default=config.skip_random_forest,
        help="Skip the Random Forest model (faster)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Reproduce the original notebook pipeline (P0), including known leakage",
    )
    args = parser.parse_args(argv)
    args.train_fraction = config.train_fraction
    args.val_fraction = config.val_fraction
    args.test_fraction = config.test_fraction
    return parser, args


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    _parser, args = build_parser(argv)
    frame = load_raw_csv(args.data, check_hash=not args.skip_hash)

    if args.legacy:
        results = run_notebook_pipeline(
            frame,
            random_state=args.seed,
            skip_random_forest=args.skip_random_forest,
        )
        print(f"Loaded {results['n_rows_raw']:,} rows; balanced to {results['n_rows_balanced']:,}")
        print(f"Train/test: {results['n_train']:,} / {results['n_test']:,}")
        _print_legacy(results)
    else:
        results = run_p1_pipeline(
            frame,
            random_state=args.seed,
            figure_dir=args.figure_dir,
            skip_random_forest=args.skip_random_forest,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
        print(f"Loaded {results['n_rows_raw']:,} rows (natural class prior)")
        _print_p1(results)
        print(f"\nWrote figures to {args.figure_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
