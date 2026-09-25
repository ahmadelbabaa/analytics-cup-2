# Who should have picked him up?

**Auditing defensive marking responsibility over time.** PySport Analytics Cup 2.0, Football, Europe edition. Deadline **18 Dec 2026**.

## The idea in one paragraph

Coaches review goals and chances asking the same question: *who should have picked him up?* We answer it objectively, for any moment of any match. Marking responsibility has no ground-truth labels, so it cannot be learned from data. It has to be **defined**, and a coach's **marking philosophy** (zonal to man-oriented) is that definition. We turn the philosophy into an assignment problem over time and **audit what the defenders actually did** against it. The tool shows:

1. who should be responsible for each attacker, frame by frame;
2. **responsibility gaps**: dangerous attackers nobody is covering in time;
3. **handover timing**: when responsibility *should* have passed from one defender to the next, versus when it did, and what the delay cost.

## Why this framing (lessons from the Cup 1 winner)

| Criterion | How we meet it |
|---|---|
| Relevance | Answers a question every defensive coach asks in video review. Gives advice ("B should have taken him 0.8 s earlier"), not an average. |
| Methodology | Marking responsibility has no labels, so it cannot be learned. It must be defined, and the coach's philosophy is the definition. Solved as an assignment problem over time: interpretable, no training data. Checked with worked examples plus population checks. |
| Originality | Responsibility **over time**: who marks whom, when marking should change hands, and the gaps when it doesn't. A **diagnosis** of what actually happened (video review), not a hypothetical better shape. |
| Communication | One **timeline** figure: responsibility lines changing hands over a sequence, ideal vs actual handover moment, and the gap in between. |
| Open-source | A small package with a clean API, built on kloppy and databallpy where possible. The grant path is a module or PR there. |

### Staying clearly distinct from the Cup 1 winner (Shah 2026, positional optimisation)

We borrow his good practice (a coaching question, a coach-controlled model, one strong figure, runs out of the box), **not his concept or look**:

- **Lead with the coaching question, never the method.** The title and first line are "Who should have picked him up?". The method is one Methods line, called an *assignment problem*; don't brand it "optimisation".
- **Time is our visual signature.** No before/after triptych of frozen positions. The main figure is a timeline of responsibility changing hands.
- **Diagnosis over prescription.** He shows where players should stand. We audit what happened in real matches: gaps and late handovers, for video review.
- **Our own methodology argument.** "No ground-truth labels for responsibility, so it must be defined by a philosophy", not "too little data for ML".
- **Cite him as complementary.** One line in the README: "Shah (2026) optimises where defenders should stand; we ask who is responsible for whom, and when that should change hands."
- **Streamlit app is optional**, a presentation aid for the final, not the centrepiece of the submission.

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
- [ ] Rewrite the README top section to the new question; keep the old plan out of it; no "optimisation" branding in title or abstract
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
- [ ] Figure 1: **timeline**, not a before/after triptych. Responsibility lines changing hands over one sequence, ideal vs actual handover moment marked, gap shaded (pitch inset allowed; colour-blind-safe palette)
- [ ] Optional figure/table 2: one small robustness or situational table (the 2-figure limit)
- [ ] `submission.ipynb`: the story, runnable top to bottom from a clean environment
- [ ] *(Optional, only if time allows)* Streamlit `app.py` for the live final: pick match and moment, set the philosophy dials, see responsibilities, gaps and handover timing

## Phase 5: submission (weeks 10–11, before 18 Dec)
- [ ] README ≤1000 words: Abstract / Introduction / Methods / Results / Conclusion; at most 2 figures/tables; reproduction steps
- [ ] README cites Shah (2026) in one line as complementary work (where to stand vs who is responsible, and when)
- [ ] Distinctness check before submitting: read the README next to the Cup 1 winner's; no shared title words, figure style or methodology sentence
- [ ] Clean-environment test: fresh venv, `pip install`, run the notebook and tests
- [ ] 1-minute YouTube pitch (script: the coach's question → one real sequence where the handover came late → the philosophy dial → "audit your own matches")
- [ ] `LICENSE.md` present; no data in repo; final check against the rules
- [ ] Submit at submissions.analytics-cup.org

## Open questions
- [ ] Threat model: xT grid (which one, licence?) or a simple goal distance/angle?
- [ ] Can SkillCorner's dynamic events (dangerous passing options, off-ball runs) be the "dangerous event" outcome without circularity?
- [x] ~~How close to the Cup 1 winner is too close?~~ Settled: same theme and good practice, different concept. See "Staying clearly distinct" above
