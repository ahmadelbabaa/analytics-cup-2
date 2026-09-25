"""Synthetic check: an attacker runs across the pitch past two static defenders -> exactly one handover A -> B."""
from handover import Match, assign_markers, detect_handovers, handover_metrics, tracking_benefit

N = 81  # attacker moves 0.25 m per frame from x=-10 to x=10


def make_match():
    # attacker 1 (team 10) runs along y=0; defenders 2 at x=-5 and 3 at x=+5 (team 20), both at y=2
    # attacker 4 stands far away so each defender has someone to mark; defender 5 marks him
    frames = {}
    for f in range(N):
        x = -10 + f * 0.25
        frames[f] = {1: (x, 0, True), 2: (-5, 2, True), 3: (5, 2, True), 4: (0, 30, True), 5: (0, 28, True)}
    return Match(1, {1: 10, 4: 10, 2: 20, 3: 20, 5: 20}, set(), 10, 20, frames, {f: 10 for f in frames})


def test_one_handover():
    m = make_match()
    h = detect_handovers(m, assign_markers(m))
    assert len(h) == 1, h
    row = h.iloc[0]
    assert (row.attacker, row.from_marker, row.to_marker) == (1, 2, 3)
    assert 44 <= row.frame <= 48  # past the midpoint (x=0, frame 40): B must be STICKY m closer, at x~1.5
    assert row.detected and row.from_distance <= 5 and row.to_distance <= 5


def test_static_defence_has_zero_benefit():
    m = make_match()
    b = tracking_benefit(m, 1, list(range(N)))
    assert abs(b.benefit).max() < 1e-9  # nobody moved, so actual == hold


def test_metrics_static_defence():
    # Static defenders 10 m apart leave a ~0.8 m dead zone around x=0 where neither is within 5 m:
    # A drops out of range before the crossover, B gets in range after it, no space leaked vs holding shape.
    m = make_match()
    q = handover_metrics(m, detect_handovers(m, assign_markers(m))).iloc[0]
    assert 0 < q.free_s < 1
    assert q.early_release_s > 0 and q.late_pickup_s > 0
    assert abs(q.space_leaked_m) < 1e-9
    assert q.vacated_gap_m > 5  # nobody covered B's zone


if __name__ == "__main__":
    test_one_handover()
    test_static_defence_has_zero_benefit()
    test_metrics_static_defence()
    print("ok")
