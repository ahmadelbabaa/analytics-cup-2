# Who should have picked him up?

**Auditing defensive marking responsibility over time.** Entry for the [PySport Analytics Cup 2.0](https://pysport.org/analytics-cup/editions/analytics-cup2/rules) (Football, Europe edition), using SkillCorner's A-League 2024/25 open data. Theme: **Defensive Positioning**. Deadline: **18 Dec 2026**.

> Work in progress. The plan and task list live in [TODO.md](TODO.md). This README becomes the <=1000-word submission write-up (max 2 figures/tables) before the deadline.

For any moment of a match, we define *who is responsible for whom* from a marking philosophy the coach chooses (zonal to man-oriented), and audit what the defenders actually did against it: responsibility gaps, and handovers that came too late.

Earlier explorations (kill tests, handover sensitivity analysis) are kept in `kill_tests/` and `analysis/` as background evidence.

## Setup

```bash
pip install -r requirements.txt
git lfs install
git clone --depth 1 https://github.com/SkillCorner/opendata.git data/skillcorner   # ~2.2 GB incl. LFS tracking
PYTHONPATH=. python tests/test_handover.py
```

```python
from handover import load_match, assign_markers, detect_handovers
m = load_match("data/skillcorner/data/matches/1874553")
handovers = detect_handovers(m, assign_markers(m))
```
