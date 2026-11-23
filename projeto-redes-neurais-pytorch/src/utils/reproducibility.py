import random
import numpy as np
import torch

from src.config import RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    """Configura as sementes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
