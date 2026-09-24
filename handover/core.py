"""Marking handovers from SkillCorner broadcast tracking.

Pipeline:
    match = load_match("data/skillcorner/data/matches/1874553")
    markers = assign_markers(match)            # who marks whom, every frame
    events = detect_handovers(match, markers)  # when marking passes from one defender to the next
    benefit = tracking_benefit(match, attacker_id, frames)  # space defenders take away vs holding shape
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MIN_HOLD = 5        # frames (0.5 s) a defender must mark an attacker to count as his marker; filters flicker
STICKY = 2.0        # metres: a marker keeps his attacker unless another defender is this much closer
MAX_DISTANCE = 5.0  # metres: only count handovers where the attacker is actually being marked (tight enough)


@dataclass
class Match:
    match_id: int
    team_of: dict        # player_id -> team_id
    goalkeepers: set
    home_team_id: int
    away_team_id: int
    frames: dict         # frame -> {player_id: (x, y, is_detected)}
    in_possession: dict  # frame -> team_id in possession (only frames where it is known)


def load_match(match_dir):
    """Read one SkillCorner open-data match folder (match json + tracking jsonl)."""
    match_dir = Path(match_dir)
    mid = int(match_dir.name)
    meta = json.loads((match_dir / f"{mid}_match.json").read_text(encoding="utf-8"))
    home, away = meta["home_team"]["id"], meta["away_team"]["id"]
    group = {"home team": home, "away team": away}
    frames, poss = {}, {}
    with open(match_dir / f"{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["player_data"]:
                continue
            frames[r["frame"]] = {p["player_id"]: (p["x"], p["y"], p["is_detected"]) for p in r["player_data"]}
            if r["possession"]["group"] in group:
                poss[r["frame"]] = group[r["possession"]["group"]]
    return Match(mid, {p["id"]: p["team_id"] for p in meta["players"]},
                 {p["id"] for p in meta["players"] if p["player_role"]["name"] == "Goalkeeper"},
                 home, away, frames, poss)


def assign_markers(match, sticky=STICKY):
    """One row per (frame, attacker): the defender marking him, by optimal 1-to-1 matching on distance.

    Sticky: the previous frame's marker gets a `sticky` metre discount, so a pairing only changes when another
    defender is clearly closer (without it, one swap reshuffles the whole assignment: 13% A->B->A ping-pong).
    Only frames where a team is in possession; goalkeepers excluded on both sides.
    ponytail: pure distance cost; add goal-side / ball-side weighting if markers look wrong on video-free checks.
    """
    rows, prev = [], {}
    for f in sorted(match.frames):
        pos = match.frames[f]
        att_team = match.in_possession.get(f)
        if att_team is None:
            prev = {}
            continue
        att = [p for p in pos if match.team_of.get(p) == att_team and p not in match.goalkeepers]
        dfn = [p for p in pos if match.team_of.get(p) not in (att_team, None) and p not in match.goalkeepers]
        if not att or not dfn:
            continue
        a = np.array([pos[p][:2] for p in att])
        d = np.array([pos[p][:2] for p in dfn])
        dist = np.linalg.norm(a[:, None] - d[None], axis=2)
        cost = dist.copy()
        for i, p in enumerate(att):
            if prev.get(p) in dfn:
                cost[i, dfn.index(prev[p])] -= sticky
        ai, di = linear_sum_assignment(cost)
        prev = {}
        for i, j in zip(ai, di):
            prev[att[i]] = dfn[j]
            rows.append((f, att[i], dfn[j], dist[i, j], pos[att[i]][2], pos[dfn[j]][2]))
    return pd.DataFrame(rows, columns=["frame", "attacker", "marker", "distance", "attacker_detected",
                                       "marker_detected"]).sort_values(["attacker", "frame"], ignore_index=True)


def _held_blocks(g):
    """Run-length blocks of one attacker's marker over contiguous frames, keeping blocks >= MIN_HOLD frames."""
    new_block = (g.marker != g.marker.shift()) | (g.frame.diff() != 1)
    b = g.assign(block=new_block.cumsum()).groupby("block").agg(
        marker=("marker", "first"), start=("frame", "first"), end=("frame", "last"))
    return b[b.end - b.start + 1 >= MIN_HOLD]


def detect_handovers(match, markers, max_gap=MIN_HOLD, max_distance=MAX_DISTANCE):
    """Handover = an attacker's held marker changes to another held marker within max_gap frames,
    while the new marker is within max_distance of him.

    Returns one row per handover: frame (first frame of the new marker), attacker, from_marker, to_marker,
    distance to the new marker, and whether attacker + both markers were detected on camera at the switch.
    """
    dist_at = markers.set_index(["frame", "attacker"]).distance
    out = []
    for att, g in markers.groupby("attacker"):
        blocks = _held_blocks(g)
        for prev, nxt in zip(blocks.itertuples(), blocks.iloc[1:].itertuples()):
            if prev.marker != nxt.marker and nxt.start - prev.end <= max_gap:
                dist = dist_at[(nxt.start, att)]
                if dist > max_distance:
                    continue
                pos = match.frames[nxt.start]
                out.append(dict(match_id=match.match_id, frame=nxt.start, attacker=att,
                                from_marker=prev.marker, to_marker=nxt.marker, distance=dist,
                                detected=bool(pos[att][2] and pos.get(prev.marker, (0, 0, False))[2]
                                              and pos[nxt.marker][2])))
    return pd.DataFrame(out, columns=["match_id", "frame", "attacker", "from_marker", "to_marker", "distance",
                                     "detected"])


def tracking_benefit(match, attacker, frames):
    """Per frame: nearest-defender separation actual vs 'hold' (defenders keep their place in the block).

    hold = each defender's position at frames[0] shifted by the block centroid's movement since then.
    benefit = hold - actual: metres of space the defence took away by moving, vs just holding shape.
    ponytail: rigid-block baseline; a per-line (defence/midfield) shift is the obvious refinement.
    """
    f0 = match.frames[frames[0]]
    team = match.team_of[attacker]
    ds = [p for p in f0 if match.team_of.get(p) not in (team, None) and p not in match.goalkeepers
          and all(p in match.frames[f] for f in frames)]
    p0 = np.array([f0[p][:2] for p in ds])
    rows = []
    for f in frames:
        p = np.array([match.frames[f][d][:2] for d in ds])
        r = np.array(match.frames[f][attacker][:2])
        actual = np.linalg.norm(p - r, axis=1).min()
        hold = np.linalg.norm(p0 + (p.mean(0) - p0.mean(0)) - r, axis=1).min()
        rows.append((f, actual, hold, hold - actual))
    return pd.DataFrame(rows, columns=["frame", "actual", "hold", "benefit"])
