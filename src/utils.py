import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt

def get_logger(name: str = "hmm"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def ensure_dir(path: str | os.PathLike):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_fig(path: str | os.PathLike):
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
