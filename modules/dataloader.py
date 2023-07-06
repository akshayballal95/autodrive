from sklearn.model_selection import train_test_split
import pandas as pd
from modules.dataset import DinoDataset
import torch


def dataloader(
    csv_file_path: str,
    test_set_split_ratio: float,
    batch_size: int,
    captures_root_dir: str,
    transform=None,
    num_workers=None,
):
    key_frame = pd.read_csv(csv_file_path)
    train, test = train_test_split(key_frame, test_size=test_set_split_ratio)
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    trainset = DinoDataset(root_dir="captures", dataframe=train, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = DinoDataset(root_dir="captures", dataframe=test, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainset, testset, trainloader, testloader
