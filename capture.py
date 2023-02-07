import shutil
import cv2
from PIL import ImageGrab
import pyautogui
import numpy as np
import keyboard
import time
import os

current_key = "1"
buffer = []

# dir = 'captures'
# for f in os.listdir(dir):
#     os.remove(os.path.join(dir, f))


def keyboardCallBack(key: keyboard.KeyboardEvent):
    global current_key

    if key.event_type == "down" and key.name not in buffer:
        buffer.append(key.name)
    
    if key.event_type == "up":
        buffer.remove(key.name)

    current_key = " ".join(buffer)

keyboard.hook(callback=keyboardCallBack)
i=0

while(1):
    time.sleep(0.1)
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (500,230,1400,450))), cv2.COLOR_RGB2BGR)
    if len(buffer)!=0:
        cv2.imwrite("captures/" +str(i)+" "+ current_key +".png", image)
    else:
         cv2.imwrite("captures/" +str(i) + " n" +".png", image)
    i= i+1
