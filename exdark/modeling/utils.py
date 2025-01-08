import os

import lightning as L
import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger


def setup_environment(seed: int):
    load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    L.seed_everything(seed)


def get_logger():
    wandb.login(key=os.environ["WANDB_TOKEN"])
    return WandbLogger(project="exdark", save_dir="wandb_artifacts_detrs")
