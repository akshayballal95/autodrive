import cv2
import torch
from torchvision.models.efficientnet import efficientnet_v2_s                                                                                          
import keyboard
from PIL import Image, ImageGrab
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from tqdm import tqdm


model = efficientnet_v2_s()
model.classifier = torch.nn.Linear(in_features = 1280, out_features = 5)
model.load_state_dict(torch.load("models/efficientnet_s_snake.pth"))
model.to("cuda")
model.eval()

label_keys= {
    0: "",
    1 :"left",
    2: "up",
    3: "right",
    4: "down"
}


transformer = Compose([
    Resize([480,480], interpolation = InterpolationMode.BILINEAR),
    CenterCrop(480),
    Normalize(mean =[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
])

def generator():
    while(not keyboard.is_pressed("esc")):
      yield

for _ in tqdm(generator()):
    image = ImageGrab.grab(bbox = (685,350,1235,840)) 

# image = Image.open("captures/2023_03_04_23_09_23.862733 right.png")
# image.show()

# image = cv2.imread("captures/2023_03_04_23_09_23.862733 right.png")
# image = Image.fromarray(image)

    image = ToTensor()(image)
    image = image.to("cuda")
    image = transformer(image)
    outputs = model(image[None , ...])
    _,preds = torch.max(outputs.data, 1)

# print(label_keys[preds.item()])
    if preds.item() != 0:
        # print(label_keys[preds.item()])
        keyboard.press_and_release(label_keys[preds.item()])





