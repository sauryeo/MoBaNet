import argparse
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "key": "base",
        "label": "BASE",
        "exp_name": "ablation_base",
        "use_cpia": False,
        "use_bagf": False,
        "mcrc": False,
    },
    {
        "key": "base_cpia",
        "label": "BASE+CPIA",
        "exp_name": "ablation_base_cpia",
        "use_cpia": True,
        "use_bagf": False,
        "mcrc": False,
    },
    {
        "key": "base_cpia_bagf",
        "label": "BASE+CPIA+BAGF",
        "exp_name": "ablation_base_cpia_bagf",
        "use_cpia": True,
        "use_bagf": True,
        "mcrc": False,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the three ablation settings for train.py."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch train.py.",
    )
    parser.add_argument(
        "--train-script",
        default="train.py",
        help="Path to the training entry script.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[exp["key"] for exp in EXPERIMENTS],
        help="Run only a subset of experiments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running later experiments after a failure.",
    )
    return parser.parse_known_args()


def bool_arg(value):
    return "true" if value else "false"


def format_command(cmd):
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return subprocess.list2cmdline(cmd)


def select_experiments(keys):
    if not keys:
        return list(EXPERIMENTS)
    selected = []
    wanted = set(keys)
    for exp in EXPERIMENTS:
        if exp["key"] in wanted:
            selected.append(exp)
    return selected


def build_command(python_exec, train_script, passthrough_args, exp):
    return [
        python_exec,
        train_script,
        *passthrough_args,
        "-mode",
        "Train",
        "-exp_name",
        exp["exp_name"],
        "-use_cpia",
        bool_arg(exp["use_cpia"]),
        "-use_bagf",
        bool_arg(exp["use_bagf"]),
        "-mcrc",
        bool_arg(exp["mcrc"]),
    ]


def main():
    args, passthrough_args = parse_args()
    experiments = select_experiments(args.only)
    train_script = str(Path(args.train_script))
    results = []

    if passthrough_args:
        print("Forwarding extra args to train.py:", " ".join(passthrough_args))

    for idx, exp in enumerate(experiments, start=1):
        cmd = build_command(args.python, train_script, passthrough_args, exp)
        print(f"\n[{idx}/{len(experiments)}] {exp['label']}")
        print(format_command(cmd))

        if args.dry_run:
            results.append((exp["label"], 0))
            continue

        completed = subprocess.run(cmd)
        results.append((exp["label"], completed.returncode))
        if completed.returncode != 0 and not args.keep_going:
            print(f"\nStopped on failure: {exp['label']} (exit code {completed.returncode})")
            return completed.returncode

    failed = [(label, code) for label, code in results if code != 0]
    print("\nSummary:")
    for label, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"- {label}: {status}")

    if failed:
        return failed[0][1]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
