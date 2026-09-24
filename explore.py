"""First look at the SkillCorner A-League 2024/25 open data, framed around defending.

Run `python explore.py` after fetching the data (see README).
"""
import collections
import glob
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "skillcorner" / "data"
pd.set_option("display.width", 200, "display.max_columns", 40)


def load_events():
    return pd.concat(
        [pd.read_csv(f, low_memory=False) for f in glob.glob(str(DATA / "matches/*/*_dynamic_events.csv"))],
        ignore_index=True,
    )


def load_tracking(match_id):
    with open(DATA / f"matches/{match_id}/{match_id}_tracking_extrapolated.jsonl") as f:
        return [json.loads(line) for line in f]


def main():
    matches = pd.json_normalize(json.load(open(DATA / "matches.json")))
    print(f"{len(matches)} matches")
    print(pd.concat([matches["home_team.short_name"], matches["away_team.short_name"]]).value_counts().to_string())

    ev = load_events()
    print(f"\n{len(ev)} dynamic events, {ev.shape[1]} columns")
    print(ev.event_type.value_counts().to_string())

    obe = ev[ev.event_type == "on_ball_engagement"]
    print("\nOn-ball engagements (defensive actions) by subtype:")
    print(obe.event_subtype.value_counts().to_string())
    print("\nEngagement outcomes (share True):")
    cols = ["stop_possession_danger", "reduce_possession_danger", "force_backward",
            "beaten_by_possession", "beaten_by_movement", "goal_side_start", "pressing_chain"]
    print(obe[cols].mean().round(3).to_string())
    print("\nPressing chains end in:", obe.pressing_chain_end_type.value_counts().to_dict())

    print("\nOut-of-possession phase (event rows):")
    print(ev.team_out_of_possession_phase_type.value_counts().to_string())

    pp = ev[ev.event_type == "player_possession"]
    print("\nDefensive structure faced by possessions (top 10):")
    print(pp.defensive_structure.value_counts().head(10).to_string())

    mid = matches.id.iloc[0]
    frames = load_tracking(mid)
    n_players = collections.Counter(len(r["player_data"]) for r in frames)
    live = [p for r in frames for p in r["player_data"]]
    print(f"\nTracking {mid}: {len(frames)} frames @10fps, {n_players[0]} without players, "
          f"{sum(p['is_detected'] for p in live) / len(live):.0%} of player positions detected (rest extrapolated)")


if __name__ == "__main__":
    main()
