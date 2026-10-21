"""Collect existing Fashion-MNIST run summaries into a comparison table."""

import argparse
import csv
import json
from pathlib import Path


def collect_runs(root: Path) -> list[dict[str, object]]:
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run = summary_path.parent
        config_path = run / "config.json"
        if not config_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "best_validation_accuracy" not in summary:
            continue  # Exclude CIFAR-10 and unrelated summaries.
        rows.append({
            "run": run.name, "seed": config.get("seed"), "epochs": config.get("epochs"),
            "learning_rate": config.get("learning_rate"), "hidden_size": config.get("hidden_size"),
            "dropout": config.get("dropout"),
            "best_validation_accuracy": summary["best_validation_accuracy"],
            "training_seconds": summary.get("training_seconds"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/week09_comparison.csv"))
    args = parser.parse_args()
    rows = collect_runs(args.results)
    if not rows:
        parser.error("no compatible runs found; train at least one src.train experiment first")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['run']}: val_acc={row['best_validation_accuracy']:.4f}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
