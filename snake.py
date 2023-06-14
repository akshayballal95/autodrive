import cv2
import torch
from torchvision.models.efficientnet import efficientnet_v2_s                                                                                          
import keyboard
from PIL import Image, ImageGrab
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from tqdm import tqdm


model = efficientnet_v2_s()
model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(in_features = 1280, out_features = 5)) 
model.load_state_dict(torch.load("models/21_adam_0.0007_35.pth"))
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
    Resize([128,128], interpolation = InterpolationMode.BILINEAR),
    CenterCrop(128),
    Normalize(mean =[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
])

def generator():
    while(not keyboard.is_pressed("esc")):
      yield

for _ in tqdm(generator()):
    image = ImageGrab.grab(bbox = (685,350,1235,840)) 

    image = ToTensor()(image)
    image = image.to("cuda")
    image = transformer(image)
    outputs = model(image[None , ...])
    preds = torch.softmax(outputs, dim=1).argmax(dim = 1)

# print(label_keys[preds.item()])
    if preds.item() != 0:
        # print(label_keys[preds.item()])
        keyboard.press_and_release(label_keys[preds.item()])



# import threading
# import time
# c = threading.Condition()
# flag = 0      #shared between Thread_A and Thread_B
# image:any

# class Thread_A(threading.Thread):
#     def __init__(self, name):
#         threading.Thread.__init__(self)
#         self.name = name

#     def run(self):
#         global flag
#         global image     #made global here
#         while True:
#             c.acquire()
#             if flag == 0:
                
#                 flag = 1
#                 image = ImageGrab.grab(bbox = (685,350,1235,840)) 
#                 c.notify_all()
#             else:
#                 c.wait()
#             c.release()


# class Thread_B(threading.Thread):
#     def __init__(self, name):
#         threading.Thread.__init__(self)
#         self.name = name

#     def run(self):
#         global flag
#         global image    #made global here
#         while True:
#             c.acquire()
#             if flag == 1:
#                 flag = 0
#                 image = ToTensor()(image)
#                 image = image.to("cuda")
#                 image = transformer(image)
#                 outputs = model(image[None , ...])
#                 preds = torch.softmax(outputs, dim=1).argmax(dim = 1)

#                 if preds.item() != 0:
#                     # print(label_keys[preds.item()])
#                     keyboard.press_and_release(label_keys[preds.item()])
#                 c.notify_all()
#             else:
#                 c.wait()
#             c.release()


# a = Thread_A("myThread_name_A")
# b = Thread_B("myThread_name_B")

# for _ in tqdm(generator()):

#     b.start()
#     a.start()

#     a.join()
#     b.join()


# from threading import Thread
# def modify_image(image):
#     image = ToTensor()(image)
#     image = image.to("cuda")
#     image = transformer(image)
#     outputs = model(image[None , ...])
#     preds = torch.softmax(outputs, dim=1).argmax(dim = 1)
#     if preds.item() != 0:
#         # print(label_keys[preds.item()])
#         keyboard.press_and_release(label_keys[preds.item()])

# for _ in tqdm(generator()):
#     image = ImageGrab.grab(bbox = (685,350,1235,840)) 

#     t = Thread(target=modify_image, args=(image, ))
#     t.start()
#     t.join()