"""Offline bootstrap only; never launches jobs or executes model tool calls."""

EXPERIMENTS = ("BASELINE", "PROMPT_CONTROL", "PATCH_REPRODUCTION", "GPU_REPLICATION")


def select_next(states, *, gpu_ready):
    """Return next pending experiment; PASS means a valid completed run, not a gain."""
    for key in states:
        if key not in EXPERIMENTS:
            return None
    for index, key in enumerate(EXPERIMENTS):
        if key not in states:
            if any(later in states for later in EXPERIMENTS[index + 1:]):
                return None
            if key == "GPU_REPLICATION" and gpu_ready is not True:
                return None
            return key
        if states[key] != "PASS":
            return None
    return None


def evaluate(expected, actual):
    """Exact fixture comparison; does not execute any tool or claim model accuracy."""
    return isinstance(expected, list) and isinstance(actual, list) and expected == actual
