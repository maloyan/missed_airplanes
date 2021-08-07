import json
import os
import sys

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import wandb
from albumentations.augmentations.geometric.resize import Resize
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import missed_planes.engine as engine
import missed_planes.metrics as metrics
from missed_planes.dataset import PlanesDataset

with open(sys.argv[1], "r") as f:
    config = json.load(f)

transforms_train = A.Compose(
    [
        A.Resize(height=config["image_size"], width=config["image_size"], p=1),
        A.OneOf(
            [
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.1),
            ],
            p=0.7,
        ),
        A.OneOf(
            [
                A.GaussNoise(p=0.5),
                A.RandomGamma(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
            ],
            p=0.3,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=180, p=0.2
        ),
    ],
    p=1,
)

transforms_test = A.Compose(
    [A.Resize(height=config["image_size"], width=config["image_size"], p=1)],
    p=1,
)
train = pd.read_csv(config["train_csv"])
test_data = pd.read_csv(config["test_csv"])

test_dataset = PlanesDataset(
    test_data, path=config["test_path"], is_test=True, augmentation=transforms_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    drop_last=False,
)
# train_data, valid_data = train_test_split(
#     train_data,
#     train_size=config["train_test_split"],
#     random_state=42,
#     stratify=train_data["sign"],
# )

skf = StratifiedKFold(n_splits=config["folds"])

for fold_num, (train_index, test_index) in enumerate(skf.split(train, train["sign"])):

    wandb.init(
        config=config,
        project=config["project"],
        name=f"{config['model']}",
    )
    train_data, valid_data = train.iloc[train_index], train.iloc[test_index]

    train_dataset = PlanesDataset(
        train_data,
        path=config["train_path"],
        is_test=False,
        augmentation=transforms_train,
    )
    valid_dataset = PlanesDataset(
        valid_data,
        path=config["train_path"],
        is_test=False,
        augmentation=transforms_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    model = eval(
        f"timm.models.{config['model']}(pretrained=True, num_classes=config['classes'])"
    )

    loss = torch.nn.BCEWithLogitsLoss()
    acc = [metrics.Accuracy()]

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=config["lr"])])

    train_epoch = engine.TrainEpoch(
        model,
        loss=loss,
        metrics=acc,
        optimizer=optimizer,
        device=config["device"],
        verbose=True,
    )

    valid_epoch = engine.ValidEpoch(
        model,
        loss=loss,
        metrics=acc,
        device=config["device"],
        verbose=True,
    )

    min_loss = 100
    patience = 0
    print("TRAINING")
    for _ in range(config["epochs"]):
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        wandb.log(
            {
                "train_loss": train_logs[type(loss).__name__],
                "val_loss": valid_logs[type(loss).__name__],
                "accuracy_train": train_logs["accuracy"],
                "accuracy_val": valid_logs["accuracy"],
            }
        )

        optimizer.param_groups[0]["lr"] *= config["decay"]
        if min_loss > valid_logs[type(loss).__name__]:
            min_loss = valid_logs[type(loss).__name__]
            torch.save(
                model,
                f"{config['checkpoint']}/fold{fold_num}_{config['model']}.pt",
            )
            print("Model saved!")
            patience = 0
        else:
            patience += 1

        if patience == config["patience"]:
            break

    model.eval()
    with torch.no_grad():
        result = []
        for i in test_loader:
            i = i.to(config["device"])

            output = model(i)
            output = output.view(-1).detach().cpu().numpy()

            result.extend(output)

    result = np.array(result).reshape(-1)

    submission = pd.read_csv("data/sample_submission.csv")
    print((result > 0.5).sum())
    submission["sign"] = result

    submission.to_csv(
        os.path.join(config["submission"], config["model"])
        + f"_fold{fold_num}"
        + ".csv",
        index=None,
    )

    submission["sign"] = (result > 0.5).astype(int)

    submission.to_csv(
        os.path.join(config["submission"], config["model"])
        + f"_fold{fold_num}"
        + ".csv.gz",
        compression="gzip",
        index=None,
    )
