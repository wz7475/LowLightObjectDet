import os

import lightning as L
import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig


def setup_environment(seed: int):
    load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    L.seed_everything(seed)
