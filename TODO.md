# Who should have picked him up?

**An optimisation tool for defensive marking responsibility.** PySport Analytics Cup 2.0, Football, Europe edition. Deadline **18 Dec 2026**.

## The idea in one paragraph

Coaches review goals and chances asking the same question: *who should have picked him up?* We answer it objectively, for any moment of any match. We solve "who is responsible for whom" as an **optimisation problem** (an assignment problem over time) under a **marking philosophy the coach chooses**, from zonal to man-oriented. We then compare that with what the defenders actually did. The tool shows:

1. who should be responsible for each attacker, frame by frame;
2. **responsibility gaps**: dangerous attackers nobody is covering in time;
3. **handover timing**: when responsibility *should* have passed from one defender to the next, versus when it did, and what the delay cost.

## Why this framing (lessons from the Cup 1 winner)

| Criterion | How we meet it |
|---|---|
| Relevance | Answers a question every defensive coach asks in video review. Gives advice ("B should have taken him 0.8 s earlier"), not an average. |
| Methodology | 20 matches is too little to *learn* marking from data. Optimisation needs no training data, is interpretable, and is controlled by the coach's philosophy. Checked with worked examples plus population checks. |
| Originality | Cup 1's winner optimised defenders' **positions** for pitch control. We optimise **responsibility over time**: who marks whom, and when marking passes on. Handover timing and responsibility gaps are new. |
| Communication | One three-panel figure (actual → ideal responsibility → gaps and late handover) and a Streamlit app with a zonal↔man dial for the live final. |
| Open-source | A small package with a clean API, built on kloppy and databallpy where possible. The grant path is a module or PR there. |

**Constraints we keep:** no team comparisons or rankings (Auckland is in 7 of 20 matches). Only players detected on camera in headline numbers. Every example is checked frame by frame before use; see the "sloppy handover" lesson below.

## Core definitions (to settle in Phase 1)

- **Attacker threat**: how much an attacker matters right now. It depends on position (xT or distance and angle to goal) and distance to the ball. Only attackers above a threshold need a responsible defender.
- **Cost of defender *d* covering attacker *a***: **time to reach** (from position and velocity, with a max speed and acceleration), not raw distance. That fixes "4 m away but already sprinting the right way." The goal-side position adds a bonus.
- **Marking philosophy (the coach's dials)**:
  - *man-orientation*: how strongly a defender keeps his current attacker (temporal stickiness, which is our existing `STICKY`);
  - *tolerance*: the allowed reach time, scaled by distance to the ball (tight near the ball, loose far away);
  - *priorities*: which attackers must always be covered (e.g. anyone in the box).
- **Ideal responsibility**: the optimal assignment under the philosophy, solved each frame and smoothed over time.
- **Actual behaviour**: which defender actually tracked the attacker, from proximity plus closing velocity. This is deliberately *not* the same optimisation, so the comparison isn't circular.
- **Responsibility gap**: a threatening attacker whose ideal defender cannot reach him within tolerance, or who has no responsible defender.
- **Handover**: the ideal responsibility for an attacker switches from A to B. **Handover delay** = when the actual tracking switched minus when the ideal switched. Its cost = gap time and separation leaked during the delay.

## Evidence plan

1. **Worked examples (the main story):** 3–4 hand-checked sequences, each shown as the three-panel figure. Pick them from typical cases, not the extreme tail.
2. **Sanity checks:** synthetic scenarios (runner crossing two zones; overload where a gap is unavoidable). As the philosophy moves from zonal to man, the number of handovers must fall and the defenders' running must rise.
3. **Supporting population numbers (one or two lines in the README):** do responsibility gaps come before dangerous events (SkillCorner's dangerous passing options, runs received, line breaks)? Carry over the existing result: handovers leak ~16% more protection than ordinary marking (sensitivity grid, 17/27 settings).
4. **Robustness:** results across philosophy settings; only players detected on camera; uncertainty from resampling whole matches.

## Lessons already learned (keep in mind)

- Kill-test headlines can be overstated. The first handover number (1.0 → 0.3 m) shrank to about 0.13 m with a fairer control group.
- Fixed radii break at distance from the ball. The "sloppy" example was really a 4 m handover in a counter-attack, 20 m from the ball. Tolerance must scale with distance to the ball.
- Undefined cases must stay undefined ("never tight"), never be filled with made-up durations.
- Extreme examples attract artifacts. Check every figure example frame by frame.

---

# TODO

## Phase 0: reset the repo (week of 28 Sep)
- [ ] Move `kill_tests/` and the current `analysis/` scripts under `analysis/appendix/`; they're background evidence, not the product
- [ ] Rewrite the README top section to the new question; keep the old plan out of it
- [ ] Decide on kloppy + databallpy as the data layer (loading, velocities); check they read the 2024/25 open data
- [ ] Load data straight from SkillCorner's GitHub URLs with a local cache, so the repo runs with no manual clone
- [ ] Add `pyproject.toml` so `pip install -e .` works; pin dependencies

## Phase 1: definitions on paper, then prototypes (weeks 1–2)
- [ ] Write down the threat, cost, philosophy and gap definitions (short `docs/definitions.md`) before coding
- [ ] Time-to-reach model: pick max speed/acceleration from the physical aggregates (e.g. PSV99) or literature; one parameter set, documented
- [ ] Attacker threat: choose xT grid vs distance/angle to goal; document why
- [ ] Synthetic test scenarios written *first* (runner crossing zones, 2v1 overload, static defence)
- [ ] Prototype the ideal assignment on 2–3 hand-picked frames; eyeball against the pitch plot

## Phase 2: build the engine (weeks 3–5)
- [ ] `responsibility(match, frames, philosophy)`: ideal assignment per frame, smoothed
- [ ] `actual_tracking(match, frames)`: who actually tracked whom (proximity + closing velocity)
- [ ] `gaps(...)`: responsibility gaps, with "never covered" kept as missing, not a number
- [ ] `handovers(...)`: ideal vs actual switch time, delay, cost (gap time, separation leaked)
- [ ] Reuse from `handover/core.py`: loading, sticky Hungarian, tracking benefit; delete what's superseded
- [ ] Tests: synthetic scenarios pass; zonal→man dial behaves monotonically (fewer handovers, more running)
- [ ] Performance: one full match in under ~30 s

## Phase 3: evidence (weeks 6–7)
- [ ] Pick 3–4 worked examples from typical cases; check each frame by frame (detection, possession, ball distance, phase)
- [ ] Population check: do gaps and late handovers come before dangerous events more than chance would suggest? Resample whole matches for uncertainty; detected-only
- [ ] Robustness across philosophy settings; one line for the README
- [ ] Situational breakdown (distance to ball, third, block type, transition vs settled). Situations only, no teams

## Phase 4: communication (weeks 8–9)
- [ ] Figure 1: three panels, actual → ideal responsibility → gaps and late handover (extend `plot_handovers.py` style, colour-blind-safe palette)
- [ ] Optional figure/table 2: one small robustness or situational table (the 2-figure limit)
- [ ] `submission.ipynb`: the story, runnable top to bottom from a clean environment
- [ ] Streamlit `app.py`: pick match and moment, set the philosophy dials, see responsibilities, gaps and handover timing

## Phase 5: submission (weeks 10–11, before 18 Dec)
- [ ] README ≤1000 words: Abstract / Introduction / Methods / Results / Conclusion; at most 2 figures/tables; reproduction steps
- [ ] Clean-environment test: fresh venv, `pip install`, run the notebook and tests
- [ ] 1-minute YouTube pitch (script: the coach's question → the dial → one example → "use it on your own matches")
- [ ] `LICENSE.md` present; no data in repo; final check against the rules
- [ ] Submit at submissions.analytics-cup.org

## Open questions
- [ ] Threat model: xT grid (which one, licence?) or a simple goal distance/angle?
- [ ] Can SkillCorner's dynamic events (dangerous passing options, off-ball runs) be the "dangerous event" outcome without circularity?
- [ ] How close to the Cup 1 winner is too close? Keep "responsibility over time" front and centre everywhere
