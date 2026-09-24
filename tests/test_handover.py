"""Synthetic check: an attacker runs across the pitch past two static defenders -> exactly one handover A -> B."""
from handover import Match, assign_markers, detect_handovers, tracking_benefit


def make_match():
    # attacker 1 (team 10) runs from x=-10 to x=10 along y=0; defenders 2 at x=-5 and 3 at x=+5 (team 20), both at y=2
    # attacker 4 stands far away so each defender has someone to mark; defender 5 marks him
    frames = {}
    for f in range(41):
        x = -10 + f * 0.5
        frames[f] = {1: (x, 0, True), 2: (-5, 2, True), 3: (5, 2, True), 4: (0, 30, True), 5: (0, 28, True)}
    return Match(1, {1: 10, 4: 10, 2: 20, 3: 20, 5: 20}, set(), 10, 20, frames, {f: 10 for f in frames})


def test_one_handover():
    m = make_match()
    h = detect_handovers(m, assign_markers(m))
    assert len(h) == 1, h
    row = h.iloc[0]
    assert (row.attacker, row.from_marker, row.to_marker) == (1, 2, 3)
    assert 21 <= row.frame <= 25  # just past the midpoint (x=0, frame 20): B must be STICKY m closer, at x~1.5
    assert row.detected


def test_static_defence_has_zero_benefit():
    m = make_match()
    b = tracking_benefit(m, 1, list(range(41)))
    assert abs(b.benefit).max() < 1e-9  # nobody moved, so actual == hold


if __name__ == "__main__":
    test_one_handover()
    test_static_defence_has_zero_benefit()
    print("ok")
