import torch


def create_model(model: torch.nn.Module, in_features, out_features, device: str):
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=in_features, out_features=out_features),
    )
    model = model.to(device)
    return model
