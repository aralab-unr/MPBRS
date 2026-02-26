#!/usr/bin/env python3
import sys
from pathlib import Path
import runpy

import humanoid_locomotion


def find_train_py() -> Path:
    """
    Locate IsaacLab's scripts/reinforcement_learning/rsl_rl/train.py.

    Works whether you run from IsaacLab root or from elsewhere via ./isaaclab.sh -p.
    """
    candidates = [
        Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py",
        Path(__file__).resolve().parent / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find scripts/reinforcement_learning/rsl_rl/train.py.\n"
        f"Tried:\n  - {candidates[0]}\n  - {candidates[1]}\n"
        "Run this from the IsaacLab repo root, or place train_n1.py in the repo root."
    )


def main() -> None:
    train_py = find_train_py()
    train_dir = train_py.parent

    # Ensure local import in train.py works: `import cli_args`
    train_dir_str = str(train_dir)
    if train_dir_str not in sys.path:
        sys.path.insert(0, train_dir_str)

    # IMPORTANT: do NOT modify sys.argv besides argv[0].
    # This preserves the exact CLI contract of train.py (argparse + hydra parsing).
    sys.argv[0] = "train.py"

    # Execute IsaacLab train entrypoint
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
