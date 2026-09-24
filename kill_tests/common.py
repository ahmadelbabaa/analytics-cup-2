import glob
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "skillcorner" / "data"


def match_dirs():
    return sorted(glob.glob(str(DATA / "matches/*")))


def load_match(match_dir):
    """events, {player_id: team_id}, {frame: {player_id: (x, y, is_detected)}} for one match."""
    mid = int(Path(match_dir).name)
    ev = pd.read_csv(f"{match_dir}/{mid}_dynamic_events.csv", low_memory=False)
    meta = json.load(open(f"{match_dir}/{mid}_match.json", encoding="utf-8"))
    team_of = {p["id"]: p["team_id"] for p in meta["players"]}
    frames = {}
    with open(f"{match_dir}/{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["player_data"]:
                frames[r["frame"]] = {p["player_id"]: (p["x"], p["y"], p["is_detected"]) for p in r["player_data"]}
    return mid, ev, team_of, frames
