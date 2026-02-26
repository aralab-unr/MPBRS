import os
import sys
import runpy

import humanoid_locomotion  # registers your Gym env in this process

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLAY_DIR = os.path.join(THIS_DIR, "scripts", "reinforcement_learning", "rsl_rl")

# Make local imports work (same trick as train)
if PLAY_DIR not in sys.path:
    sys.path.insert(0, PLAY_DIR)

# Forward args to IsaacLab play.py (or eval.py depending on your IsaacLab version)
sys.argv = ["play.py"] + sys.argv[1:]

runpy.run_path(os.path.join(PLAY_DIR, "play.py"), run_name="__main__")

