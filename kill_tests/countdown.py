"""Kill test for "The Countdown": are there enough clean, defender-caused deaths of dangerous passing options?

A dangerous option "dies" if it ends while the carrier still has the ball; otherwise it is censored
(carrier passed / possession ended). For each option we compare tracking at its start and end frame.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data" / "skillcorner" / "data"
DEATH_MARGIN = 3     # frames: option must end at least 0.3 s before the possession ends to count as a death
CLOSE_M = 1.0        # ponytail: crude cause rule, nearest defender moved >=1 m closer to the lane -> "defender closed"


def lane_distance(p, a, b):
    """Distance from point(s) p to segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def snapshot(frame, carrier, receiver, defenders):
    """Positions + detection flags we need from one tracking frame, or None if a key player is missing."""
    pos = {p["player_id"]: p for p in frame["player_data"]}
    if carrier not in pos or receiver not in pos:
        return None
    a = np.array([pos[carrier]["x"], pos[carrier]["y"]])
    b = np.array([pos[receiver]["x"], pos[receiver]["y"]])
    d_ids = [d for d in defenders if d in pos]
    if not d_ids:
        return None
    dxy = np.array([[pos[d]["x"], pos[d]["y"]] for d in d_ids])
    dist = lane_distance(dxy, a, b)
    i = int(dist.argmin())
    return dict(lane=dist[i], defender=d_ids[i], def_detected=pos[d_ids[i]]["is_detected"],
                rec_detected=pos[receiver]["is_detected"], car_detected=pos[carrier]["is_detected"],
                gap=np.linalg.norm(b - a), pos=pos)


def match_rows(match_dir):
    mid = int(Path(match_dir).name)
    ev = pd.read_csv(f"{match_dir}/{mid}_dynamic_events.csv", low_memory=False)
    meta = json.load(open(f"{match_dir}/{mid}_match.json", encoding="utf-8"))
    team_of = {p["id"]: p["team_id"] for p in meta["players"]}

    pp_end = ev[ev.event_type == "player_possession"].set_index("event_id").frame_end
    po = ev[(ev.event_type == "passing_option") & (ev.dangerous == True)].copy()
    po["pp_end"] = po.associated_player_possession_event_id.map(pp_end)
    po["died"] = po.frame_end <= po.pp_end - DEATH_MARGIN

    need = set(po.frame_start) | set(po.frame_end)
    frames = {}
    with open(f"{match_dir}/{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["frame"] in need:
                frames[r["frame"]] = r

    out = []
    for o in po.itertuples():
        carrier, receiver = int(o.player_in_possession_id), int(o.player_id)
        defenders = [p for p, t in team_of.items() if t != team_of.get(receiver)]
        s = snapshot(frames[o.frame_start], carrier, receiver, defenders) if o.frame_start in frames else None
        e = snapshot(frames[o.frame_end], carrier, receiver, defenders) if o.frame_end in frames else None
        if s is None or e is None:
            out.append(dict(match_id=mid, died=o.died, usable=False))
            continue
        # follow the defender who is nearest the lane at the END (the candidate "closer")
        dz = e["defender"]
        a0 = np.array([s["pos"][carrier]["x"], s["pos"][carrier]["y"]])
        b0 = np.array([s["pos"][receiver]["x"], s["pos"][receiver]["y"]])
        lane_start_same = (lane_distance(np.array([[s["pos"][dz]["x"], s["pos"][dz]["y"]]]), a0, b0)[0]
                           if dz in s["pos"] else np.nan)
        out.append(dict(
            match_id=mid, died=o.died, usable=True, duration=o.duration, xthreat=o.xthreat,
            lane_start=lane_start_same, lane_end=e["lane"], d_lane=e["lane"] - lane_start_same,
            d_gap=e["gap"] - s["gap"],
            clean=bool(e["def_detected"] and s["pos"].get(dz, {}).get("is_detected", False)
                       and s["rec_detected"] and e["rec_detected"]),
        ))
    return out


def main():
    rows = [r for d in sorted(glob.glob(str(DATA / "matches/*"))) for r in match_rows(d)]
    df = pd.DataFrame(rows)
    print(f"dangerous options: {len(df)}   died during possession: {df.died.sum()}   censored: {(~df.died).sum()}")
    u = df[df.usable]
    print(f"usable (carrier, receiver, defenders in tracking at start+end): {len(u)}")
    c = u[u.clean]
    print(f"clean (closing defender + receiver detected at start AND end): {len(c)}  "
          f"-> deaths {c.died.sum()}, censored {(~c.died).sum()}")

    deaths = c[c.died]
    closed = deaths.d_lane <= -CLOSE_M
    print(f"\nclean deaths with defender closing >= {CLOSE_M} m on the lane: {closed.sum()} "
          f"({closed.mean():.0%} of clean deaths)")
    print(f"clean deaths where receiver-carrier gap grew > 3 m (attacker drifting away): {(deaths.d_gap > 3).sum()}")

    print("\nSignal check - lane distance change (m), deaths vs censored:")
    print(c.groupby("died").d_lane.describe()[["count", "mean", "25%", "50%", "75%"]].round(2))
    print("\ndeaths per match (clean, defender-closed):")
    print(deaths[closed].groupby("match_id").size().describe().round(1).to_dict())
    df.to_csv(Path(__file__).parent.parent / "data" / "kill_test_options.csv", index=False)


if __name__ == "__main__":
    main()
