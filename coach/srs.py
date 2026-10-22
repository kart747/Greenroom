"""Spaced repetition with SM-2 (the SuperMemo-2 algorithm, as used by Anki).

Idea: review a word just before you would forget it. Each successful review
pushes the next one further out (1 day, 6 days, then x ease), and the "ease"
factor learns how hard this particular card is for this learner.

Grades (the four buttons):
    0  Again  forgot it      -> start over, see it again in 10 minutes
    3  Hard   remembered with effort
    4  Good   remembered
    5  Easy   remembered instantly
Ease starts at 2.5 and never drops below 1.3 (below that, cards would come
back so often that reviewing becomes a chore).
"""

from __future__ import annotations

import time

DAY = 86_400
RELEARN_DELAY = 10 * 60   # "Again": back in the same practice sitting, not tomorrow
GRADES = {0: "again", 3: "hard", 4: "good", 5: "easy"}
MIN_EASE = 1.3


def review(ease: float, interval: int, reps: int, grade: int,
           now: float | None = None) -> tuple[float, int, int, float]:
    """Apply one review. Returns (ease, interval_days, reps, due_time)."""
    if grade not in GRADES:
        raise ValueError(f"grade must be one of {sorted(GRADES)}")
    now = time.time() if now is None else now

    # SM-2 ease update: 5 raises it, 4 keeps it, 3 lowers it, 0 lowers it a lot.
    ease = max(MIN_EASE, ease + 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))

    if grade < 3:
        return round(ease, 3), 0, 0, now + RELEARN_DELAY
    reps += 1
    if reps == 1:
        interval = 1
    elif reps == 2:
        interval = 6
    else:
        interval = round(interval * ease)
    return round(ease, 3), interval, reps, now + interval * DAY
