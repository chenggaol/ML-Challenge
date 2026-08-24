"""Sequential hyperparameter sweep over train.py.

parameters swept one group at a time, within group the highest macro-f1 score moves on
greedy coordinate search, not full grid

parameters which are swept from important to least important:
encoder arch(cnn, transformer), projector arch(linear,mlp), lr(3e-4,1e-2,1e-1)
noise(true,false), batch_size(16,64), scheduler type(cosine,step)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# (train.py flag name, candidate values). "projector" only applies to --model context.
# batch_size=1 is intentionally left out: it breaks BatchNorm1d (used by the CNN
# encoder) in train mode, since batch statistics need more than one sample.
SWEEP_STAGES = [
    ("encoder", ["cnn", "transformer"]),
    ("projector", ["linear", "mlp"]),
    ("lr", ["3e-4", "1e-2", "1e-1"]),
    ("augment", [False, True]),
    ("batch_size", ["16", "64"]),
    ("lr_scheduler", ["cosine", "step"]),
]

VAL_F1_PATTERN = re.compile(r"saved best model \(val_f1=([\d.]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy sequential hyperparameter sweep over train.py")
    parser.add_argument("--model", choices=["direct", "context"], default="context")
    parser.add_argument("--epochs", type=int, default=15, help="epochs per sweep trial (keep short; use more for the final run)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_cli_args(name: str, model: str, epochs: int, seed: int, params: dict) -> list[str]:
    cli_args = ["--name", name, "--model", model, "--epochs", str(epochs), "--seed", str(seed)]
    for key, value in params.items():
        if key == "augment":
            if value:
                cli_args.append("--augment")
            continue
        cli_args += [f"--{key}", str(value)]
    return cli_args


def run_trial(name: str, model: str, epochs: int, seed: int, params: dict) -> float:
    cli_args = [sys.executable, "train.py", *build_cli_args(name, model, epochs, seed, params)]
    result = subprocess.run(cli_args, capture_output=True, text=True, cwd=ROOT)

    match = VAL_F1_PATTERN.search(result.stdout)
    if match is None:
        print("---- trial failed, stdout ----")
        print(result.stdout[-3000:])
        print("---- trial failed, stderr ----")
        print(result.stderr[-3000:])
        raise RuntimeError(f"run '{name}' did not report a val_f1 (exit code {result.returncode})")

    return float(match.group(1))


def main() -> None:
    args = parse_args()
    best_params: dict = {}

    for stage_name, candidates in SWEEP_STAGES:
        if stage_name == "projector" and args.model == "direct":
            print(f"[{stage_name}] skipped: direct model has no projector\n")
            continue

        print(f"[{stage_name}] sweeping {candidates}")
        best_value = None
        best_f1 = -1.0

        for candidate in candidates:
            trial_params = {**best_params, stage_name: candidate}
            run_name = f"sweep_{stage_name}_{candidate}"

            print(f"  {stage_name}={candidate} ...", end=" ", flush=True)
            val_f1 = run_trial(run_name, args.model, args.epochs, args.seed, trial_params)
            print(f"val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_value = candidate

        best_params[stage_name] = best_value
        print(f"[{stage_name}] best = {best_value} (val_f1={best_f1:.4f})\n")

    print("Final selected hyperparameters (chosen on validation macro-F1):")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    final_cmd = ["python", "train.py"] + build_cli_args("final", args.model, args.epochs, args.seed, best_params)
    print("\nSuggested final run with these hyperparameters (bump --epochs up for the real run):")
    print("  " + " ".join(final_cmd))


if __name__ == "__main__":
    main()
