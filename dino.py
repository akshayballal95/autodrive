import torch
from torchvision.models.efficientnet import efficientnet_v2_m
import pyautogui
import keyboard
import cv2
from PIL import Image, ImageGrab
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

model = efficientnet_v2_m()
model.classifier = torch.nn.Linear(in_features = 1280, out_features = 2)
model.load_state_dict(torch.load("efficientnet.pth", map_location=torch.device('cpu')))
model.eval()

transformer = Compose([
    Resize(480),
    CenterCrop(480),
    ToTensor()
])

while(not keyboard.is_pressed("esc")):
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (500,230,1400,450))), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize = (480,480))
    image = Image.fromarray(image)
    image = transformer(image)
    image = image.reshape(1,3,480,480)
    outputs = model(image)
    _,preds = torch.max(outputs.data, 1)
    print(preds.item())
    if preds == 1:
        keyboard.press_and_release()


