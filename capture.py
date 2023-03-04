import cv2
from PIL import ImageGrab
import numpy as np
import keyboard
import os

current_key = "1"
buffer = []


isExist = os.path.exists("captures")

if isExist:
    dir = 'captures'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

else:

    os.mkdir("captures")

def keyboardCallBack(key: keyboard.KeyboardEvent):
    global current_key

    if key.event_type == "down" and key.name not in buffer:
        buffer.append(key.name)
    
    if key.event_type == "up":
        buffer.remove(key.name)

    current_key = " ".join(buffer)

keyboard.hook(callback=keyboardCallBack)
i=0

while(not keyboard.is_pressed("esc")):
    
    ###uncomment for dino game
    # image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (620,220,1280,360))), cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (685,350,1235,840))), cv2.COLOR_RGB2BGR)
    if len(buffer)!=0:
        cv2.imwrite("captures/" +str(i)+" "+ current_key +".png", image)
    else:
         cv2.imwrite("captures/" +str(i) + " n" +".png", image)
    i= i+1
