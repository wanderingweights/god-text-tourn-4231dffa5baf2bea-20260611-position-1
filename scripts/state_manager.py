import json
import os

STATE_FILE = "/tmp/trainer_state.json"


def get_state() -> dict:
    """Return the current training state from file."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def set_state(state: dict) -> None:
    """Save the training state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
