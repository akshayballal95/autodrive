import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from engine import train_step, test_step
from models import create_model
from dataloader import dataloader
import os
import datetime

BATCH_SIZE = 32
EPOCHS = 10

transformer = Compose([
    Resize((128,128)),
    CenterCrop(128),
    ToTensor(),
    Normalize(mean =[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225] )
])

def create_writer(experiment_name: str, model_name:str, optimizer_name:str, learning_rate,  extra:str="") -> SummaryWriter:
    timestamp= str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    learning_rate = str(learning_rate)
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name, optimizer_name,learning_rate, extra).replace("\\", "/")
    return SummaryWriter(log_dir=log_dir)

# Loader
trainloader, testloader = dataloader(csv_file_path="labels.csv", test_set_split_ratio=0.2,
                                     captures_root_dir="captures", transform=transformer, batch_size=BATCH_SIZE, num_workers=2)

# Create Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_model(model=torchvision.models.efficientnet_v2_s(
), in_features=1280, out_features=5, device=device, )

criterion = torch.nn.CrossEntropyLoss()


# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer_names = ["adam", "sgd"]
learning_rates = [0.001, 0.003, 0.005, 0.007, 0.009]

experiment_number = 0
for optimizer_name in optimizer_names:
    for learning_rate in learning_rates:
        experiment_number += 1
        print("Experiment Number: ", experiment_number)
        print("Optimizer: ", optimizer_name, " Learning Rate: ", learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name == "adam" else optim.SGD(
            model.parameters(), lr=learning_rate)
        writer = create_writer(experiment_name=f'{experiment_number}', model_name = "effnet_v2_s",
                               optimizer_name=optimizer_name, learning_rate=learning_rate)

        for epoch in range(EPOCHS):
            train_loss, accuracy = train_step(
                model=model, optimizer=optimizer, criterion=criterion, trainloader=trainloader, epoch=0, epoch_number=EPOCHS)
            writer.add_scalar(tag = "Training Loss", scalar_value=train_loss, global_step = epoch)
            writer.add_scalar(tag = "Training Accuracy", scalar_value=accuracy, global_step = epoch)

        PATH = f'models/{experiment_number}_{optimizer_name}_{learning_rate}'

        print("Saving model")

        test_loss, test_acc = test_step(model=model, criterion=criterion, testloader=testloader, epoch=0)

        torch.save(model.state_dict(), PATH)

