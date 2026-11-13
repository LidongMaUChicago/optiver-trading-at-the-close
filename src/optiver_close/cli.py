# Thread limits and the torch-before-xgboost import order are set in __init__.
import argparse
import json
from pathlib import Path
from .experiment import run_experiment


def main():
    p = argparse.ArgumentParser(description="Optiver closing-auction forecasting research")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/quick.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-full-run", action="store_true", help="Explicitly enable long full-data experiments")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if config["run_type"] == "full_research" and not a.allow_full_run:
        p.error("Full training can take hours; pass --allow-full-run explicitly")
    report = run_experiment(a.data, config, a.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

