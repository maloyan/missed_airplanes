import json
import os
import sys

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import ttach as tta
from albumentations.augmentations.geometric.resize import Resize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import missed_planes.engine as engine
import missed_planes.metrics as metrics
from missed_planes.dataset import PlanesDataset

with open(sys.argv[1], "r") as f:
    config = json.load(f)

transforms = A.Compose(
    [
        A.Resize(height=config["image_size"], width=config["image_size"], p=1),
    ],
    p=1,
)

test_data = pd.read_csv(config["test_csv"])

test_dataset = PlanesDataset(
    test_data, path=config["test_path"], is_test=True, augmentation=transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    drop_last=False,
)

with torch.no_grad():
    final = []
    for ind in range(config["folds"]):
        model = torch.load(f"{config['checkpoint']}/fold{ind}_{config['model']}.pt")
        model.eval()
        tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    
        result = []
        for i in tqdm(test_loader, total=len(test_loader)):
            i = i.to(config["device"])

            output = tta_model(i)
            output = output.view(-1).detach().cpu().numpy()

            result.extend(output)

        final.append(result)

result = np.array(final).mean(axis=0)

submission = pd.read_csv("data/sample_submission_extended.csv")
submission["sign"] = result
# import IPython; IPython.embed(); exit(1)
# submission["sign"] = (result > 0.5).astype(int)
# print((result > 0.5).sum())

submission.to_csv(
    os.path.join(config["submission"], config["model"]) + ".csv",
    index=None,
)

submission.to_csv(
    os.path.join(config["submission"], config["model"]) + ".csv.gz",
    compression="gzip",
    index=None,
)

