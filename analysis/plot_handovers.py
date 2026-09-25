"""Pitch visual: a clean handover next to a sloppy one (A releases early / B picks up late, runner left free).

Needs data/defensive_metrics.csv from analysis/defensive_metrics.py.
Run from the repo root:  PYTHONPATH=. python analysis/plot_handovers.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from handover import load_match

DATA = Path("data/skillcorner/data/matches")
OUT = Path("figures/handover_examples.png")
W, RADIUS = 20, 3.0
# reference palette, first three categorical slots (validated all-pairs) + neutrals
RUNNER, A_COL, B_COL = "#eb6834", "#2a78d6", "#1baf7a"
INK, MUTED, SURFACE = "#0b0b0b", "#8a8984", "#fcfcfb"


def runner_travel(match, row):
    p0, p1 = match.frames[row.frame - W][row.attacker], match.frames[row.frame + W][row.attacker]
    return np.hypot(p1[0] - p0[0], p1[1] - p0[1])


def pick_examples(df):
    """Most readable clean and sloppy examples: runner travels far, action away from the touchline."""
    clean = df[(df.early_release_s == 0) & (df.late_pickup_s == 0) & (df.free_s == 0)]
    sloppy = df[(df.early_release_s > 0) & (df.late_pickup_s > 0) & (df.free_s >= 1.5)]
    return clean, sloppy


def ball_at(match_dir, frame):
    mid = match_dir.name
    with open(match_dir / f"{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["frame"] == frame:
                return r["ball_data"]["x"], r["ball_data"]["y"]
    return None


def draw(ax, pitch, match, match_dir, row, title):
    fs = range(row.frame - W, row.frame + W + 1)
    defending = {p for p, t in match.team_of.items() if t != match.team_of[row.attacker]}
    gk = [p for p in match.frames[row.frame] if p in match.goalkeepers and p in defending]
    flip = -1 if gk and match.frames[row.frame][gk[0]][0] < 0 else 1   # attack left -> right
    xy = lambda f, p: (flip * match.frames[f][p][0], flip * match.frames[f][p][1])

    pitch.draw(ax=ax)
    pos = match.frames[row.frame]
    for p in pos:
        if p in (row.attacker, row.from_marker, row.to_marker):
            continue
        x, y = xy(row.frame, p)
        if p in defending:
            ax.scatter(x, y, s=70, color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        else:
            ax.scatter(x, y, s=70, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.5, zorder=3)
    ball = ball_at(match_dir, row.frame)
    if ball:
        ax.scatter(flip * ball[0], flip * ball[1], s=30, color=INK, zorder=6)

    run = np.array([xy(f, row.attacker) for f in fs])
    defenders = [p for p in defending if p not in match.goalkeepers]
    free = np.array([min(np.hypot(*(np.array(xy(f, d)) - np.array(xy(f, row.attacker))))
                         for d in defenders if d in match.frames[f]) > RADIUS for f in fs])
    # unmarked stretch of the run: a wide translucent band under the runner's trail
    for i in range(len(run) - 1):
        if free[i] and free[i + 1]:
            ax.plot(run[i:i + 2, 0], run[i:i + 2, 1], color=RUNNER, lw=9, alpha=0.25,
                    solid_capstyle="round", zorder=4)

    ends = []
    for pid, col, name in [(row.attacker, RUNNER, "Runner"), (row.from_marker, A_COL, "A (hands over)"),
                           (row.to_marker, B_COL, "B (takes over)")]:
        tr = np.array([xy(f, pid) for f in fs])
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=2, zorder=5)
        ax.scatter(*tr[0], s=40, facecolor=SURFACE, edgecolor=col, linewidth=2, zorder=6)
        ax.scatter(*tr[-1], s=110, color=col, edgecolor=SURFACE, linewidth=2, zorder=7)
        ends.append((tr[-1], name))
    # labels stacked by end height so close end points never collide: lowest below, highest above,
    # middle on whichever side faces away from the other two end points
    lo, mid, hi = sorted(ends, key=lambda e: e[0][1])
    right = mid[0][0] >= (lo[0][0] + hi[0][0]) / 2
    for (pt, name), (dx, dy, ha, va) in [(lo, (0, -12, "center", "top")), (hi, (0, 12, "center", "bottom")),
                                         (mid, (10, 0, "left", "center") if right else (-10, 0, "right", "center"))]:
        ax.annotate(name, pt, xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK,
                    ha=ha, va=va, zorder=8)
    sx, sy = xy(row.frame, row.attacker)
    ax.scatter(sx, sy, s=160, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=7)
    ax.annotate("switch", (sx, sy), xytext=(-8, -14), textcoords="offset points", fontsize=8, color=MUTED,
                ha="right")

    cx, cy = run.mean(0)
    ax.set_xlim(max(-54, cx - 24), min(54, cx + 24))
    ax.set_ylim(max(-36, cy - 16), min(36, cy + 16))
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.text(0.0, -0.06, f"Unmarked (no defender within {RADIUS:.0f} m): {row.free_s:.1f} s   ·   "
            f"A released early: {row.early_release_s:.1f} s   ·   B picked up late: {row.late_pickup_s:.1f} s",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)


def main():
    df = pd.read_csv("data/defensive_metrics.csv")
    clean, sloppy = pick_examples(df)
    picks = []
    for cand in (clean, sloppy):
        best, best_row = -1, None
        for mid in cand.match_id.unique()[:6]:           # ponytail: scan a few matches, enough to find a clear case
            m = load_match(DATA / str(mid))
            for row in cand[cand.match_id == mid].itertuples():
                t = runner_travel(m, row)
                if t > best and abs(m.frames[row.frame][row.attacker][1]) < 25:
                    best, best_row, best_m = t, row, m
        picks.append((best_m, best_row))

    pitch = Pitch(pitch_type="skillcorner", pitch_length=105, pitch_width=68, line_color="#c9c8c2",
                  pitch_color=SURFACE, linewidth=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE)
    titles = ["Clean handover: B takes the runner before A lets go",
              "Sloppy handover: A drops off, B arrives late, runner is free"]
    for ax, (m, row), title in zip(axes, picks, titles):
        draw(ax, pitch, m, DATA / str(m.match_id), row, title)
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", markersize=8, label=l) for c, l in
               [(RUNNER, "Runner (attacker)"), (A_COL, "Defender A: hands the runner over"),
                (B_COL, "Defender B: takes the runner over")]]
    handles += [plt.Line2D([], [], color=RUNNER, lw=9, alpha=0.25, label="Runner unmarked"),
                plt.Line2D([], [], ls="", marker="o", markersize=8, color=MUTED, label="Other defenders"),
                plt.Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=SURFACE,
                           markeredgecolor=MUTED, label="Other attackers")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Marking handovers: 4 s of tracking around the switch (hollow = start, filled = end), "
                 "attack left to right", fontsize=12, color=INK, x=0.012, y=0.985, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(OUT, [(m.match_id, int(r.frame)) for m, r in picks])


if __name__ == "__main__":
    main()
