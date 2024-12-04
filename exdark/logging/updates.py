import os

import wandb
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.environ["WANDB_TOKEN"])
api = wandb.Api()
entity = "wzwz7475-warsaw-university-of-technology-"
project = "exdark"
run_id = "7dt0fat5"

# Get the run
run = api.run(f"{entity}/{project}/{run_id}")
params = {
    "augmentations": "HorizontalFlip - 0.5; RandomBrightnessContrast - 0.2; A.Perspective - 0.1, HueSaturationValue 0.1"
}
for param in params:
    run.config[param] = params[param]
run.update()