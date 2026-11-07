"""Explicit existing optimization policies, independent of coordinator plumbing."""


def carries_optimizer(c):
    return c.get("workflow") in ("fresh_real", "completion_arm")
