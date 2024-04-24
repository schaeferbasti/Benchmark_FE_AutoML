from openfe import OpenFE, transform

from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import pandas as pd

import torch
from torch import nn, optim
import torch.nn.functional as f
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from amltk import Node, Trial
from amltk import Choice, Component, Metric, Sequential
from amltk.optimization.optimizers.smac import SMACOptimizer
from lib.builders import (MatchChosenDimensions, MatchDimensions, build_model_from_pipeline)


def get_openFE_features(train_x, test_x, train_y, n_jobs):
    openFE = OpenFE()
    features = openFE.fit(data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
    return train_x, test_x


def calc_openFE_score(train_x, test_x, train_y, test_y):
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(test_x, test_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    return score


def model_eval(
        model: nn.Module,
        device: torch.device,
        test_loader: torch.utils.data.DataLoader,
) -> tuple[float, float]:
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for _test_data, _test_target in test_loader:
            test_data, test_target = _test_data.to(device), _test_target.to(device)
            output = model(test_data)
            test_loss += f.nll_loss(output, test_target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return float(test_loss), float(accuracy)


def eval_configuration(
        trial: Trial,
        pipeline: Node,
        device: str = "cpu",  # Change if you have a GPU
        epochs: int = 1,  # Fixed for now
        lr: float = 0.1,  # Fixed for now
        gamma: float = 0.7,  # Fixed for now
        batch_size: int = 64,  # Fixed for now
        log_interval: int = 10,  # Fixed for now
) -> Trial.Report:
    trial.store({"config.json": pipeline.config})
    torch.manual_seed(trial.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    _device = torch.device(device)
    print("Using device", _device)

    model = (
        pipeline.configure(trial.config)
        .build(builder=build_model_from_pipeline)
        .to(_device)
    )

    with trial.profile("training"):
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(epochs):
            for batch_idx, (_data, _target) in enumerate(train_loader):
                optimizer.zero_grad()
                data, target = _data.to(_device), _target.to(_device)

                output = model(data)
                loss = f.nll_loss(output, target)

                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        ),
                    )
                    lr_scheduler.step()

    final_train_loss, final_train_acc = model_eval(model, _device, train_loader)
    final_test_loss, final_test_acc = model_eval(model, _device, test_loader)
    trial.summary["final_test_loss"] = final_test_loss
    trial.summary["final_test_accuracy"] = final_test_acc
    trial.summary["final_train_loss"] = final_train_loss
    trial.summary["final_train_accuracy"] = final_train_acc

    return trial.success(accuracy=final_test_acc)


def def_pipeline():
    pipeline_to_return = Sequential(
        Choice(
            Sequential(
                nn.Flatten(start_dim=1),
                Component(
                    nn.Linear,
                    config={"in_features": 784, "out_features": 100},
                    name="choice1-fc1",
                ),
                name="choice1",
            ),
            Sequential(
                Component(
                    nn.Conv2d,
                    config={
                        "in_channels": 1,  # MNIST images are grayscale
                        "out_channels": 32,  # Number of output channels (filters)
                        "kernel_size": (3, 3),  # Size of the convolutional kernel
                        "stride": (1, 1),  # Stride of the convolution
                        "padding": (1, 1),  # Padding to add to the input
                    },
                    name="choice2",
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Flatten(start_dim=1),
                name="choice2",
            ),
            name="layer1",
        ),
        Component(
            nn.Linear,
            config={
                "in_features": MatchChosenDimensions(
                    choice_name="layer1",
                    choices={"choice1": 100, "choice2": 32 * 14 * 14},
                ),
                "out_features": MatchDimensions("fc2", param="in_features"),
            },
            name="fc1",
        ),
        Choice(nn.ReLU(), nn.Sigmoid(), name="activation"),
        Component(
            nn.Linear,
            space={"in_features": (10, 50), "out_features": (10, 30)},
            name="fc2",
        ),
        Component(
            nn.Linear,
            config={
                "in_features": MatchDimensions("fc2", param="out_features"),
                "out_features": 10,
            },
            name="fc3",
        ),
        Component(nn.LogSoftmax, config={"dim": 1}),
        name="my-mlp-pipeline",
    )
    return pipeline_to_return

if __name__ == "__main__":
    # OpenFE
    n_jobs = 4

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.device(_device)

    # Download the dataset
    """
    train_data = datasets.MNIST("../data", train=True, download=False)
    test_data = datasets.MNIST("../data", train=False, download=False)
    train_x = train_data.train_data
    train_y = train_data.train_labels
    test_x = test_data.test_data
    test_y = test_data.test_labels
    """

    data = fetch_california_housing(as_frame=True).frame
    label = data[['MedHouseVal']]

    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
    print(train_x.head(5))

    train_x, test_x = get_openFE_features(train_x, test_x, train_y, n_jobs)
    print(train_x.head(5))
    score = calc_openFE_score(train_x, test_x, train_y, test_y)
    print("Score: " + str(score))

    # AutoML-Toolkit
    pipeline = def_pipeline()
    metric = Metric("accuracy", minimize=False, bounds=(0, 1))
    optimizer = SMACOptimizer.create(
        space=pipeline,
        metrics=metric,
        seed=1,
        bucket="pytorch-experiments",
    )

    # Iteratively evaluate different configurations using the optimizer
    trial = optimizer.ask()
    report = eval_configuration(trial, pipeline, device=_device)
    optimizer.tell(report)
    print(report)