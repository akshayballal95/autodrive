import os
import shutil
import numpy as np
import csv 

labels = []

dir = 'captures'
for f in os.listdir(dir):
    key = f.rsplit('.',1)[0].rsplit(" ",1)[1]
    
    if key=="n":
        labels.append({'file_name': f, 'class': 0})
    elif key=="left":
        labels.append({'file_name': f, 'class': 1})
    elif key=="up":
        labels.append({'file_name': f, 'class': 2})
    elif key=="right":
        labels.append({'file_name': f, 'class': 3})
    elif key=="down":
        labels.append({'file_name': f, 'class': 4})
    

field_names= ['file_name', 'class']

print(labels)

with open('labels_snake.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(labels)
