"""The one exception for optional features.

Voice, BERT ranking and grammar correction all depend on models that a fresh
checkout may not have downloaded yet. Instead of crashing, each loader raises
Unavailable with a message written for the learner ("Voice off: run setup").
The app catches it, turns that feature off, and text chat keeps working.
"""


class Unavailable(RuntimeError):
    """A feature can't run here; str(err) says how to fix it."""
