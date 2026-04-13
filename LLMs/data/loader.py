import json
import os

from datasets import load_dataset

from config.settings import Config


def load_hf_dataset(cfg: Config):
    """Load the HuggingFace dataset, authenticating via HF_TOKEN if set."""
    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    return load_dataset(cfg.hf_dataset, split=cfg.hf_split, token=token)


def load_already_done(results_file: str) -> set[int]:
    """Return the set of record indices already written to the results file."""
    done = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    done.add(record["index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done