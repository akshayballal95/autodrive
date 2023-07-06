import torch
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(
    model, trainloader, optimizer, criterion, epoch, epoch_number, device=device
):
    running_loss = 0.0
    accuracy = 0.0
    pbar = tqdm(trainloader, position=0, leave=True, colour="green", ncols=100)
    for data in pbar:
        pbar.set_description("Epoch: {}/{}".format(epoch + 1, epoch_number))
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        outputs = model(inputs)
        y_pred = torch.softmax(outputs, dim=1).argmax(dim=1)
        accuracy += (y_pred == labels).sum() / len(labels)

        optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss
    tqdm.write("Training Accuracy: " + str(accuracy / len(trainloader)))
    # tqdm.write("Current Loss: " + str(running_loss/len(trainloader)))
    return running_loss / len(trainloader), accuracy / len(trainloader)


def test_step(model, testloader, criterion, device=device):
    running_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        pbar = tqdm(testloader, position=0, leave=True, colour="green", ncols=100)
        for data in pbar:
            pbar.set_description("Testing")
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            outputs = model(inputs)
            y_pred = torch.softmax(outputs, dim=1).argmax(dim=1)
            accuracy += (y_pred == labels).sum() / len(labels)

            loss = criterion(outputs, labels)
            running_loss += loss
    tqdm.write("Testing Accuracy: " + str(accuracy / len(testloader)))
    return running_loss / len(testloader), accuracy / len(testloader)
