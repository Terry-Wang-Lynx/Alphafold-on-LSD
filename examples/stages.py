"""Small configurable three-stage interface demonstration, not a tuned recipe."""
from hallucination_lab import Stage


def make_stages(config):
    counts = config.get("stage_steps", [4, 4, 2])
    if (not isinstance(counts, list) or len(counts) != 3
            or any(type(n) is not int or n < 1 for n in counts)):
        raise ValueError("stage_steps must be three positive integers")
    lr = config.get("learning_rate", 0.05)
    return [Stage("logits", counts[0], lr,
                  {"soft": 0.0, "hard": 0.0, "temperature": 1.0}, {"soft": 1.0}),
            Stage("soft", counts[1], lr,
                  {"soft": 1.0, "hard": 0.0, "temperature": 1.0}, {"temperature": 0.2}),
            Stage("hard", counts[2], lr,
                  {"soft": 1.0, "hard": 1.0, "temperature": 0.2})]
