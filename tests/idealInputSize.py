from PIL import Image, ImageDraw
import torch 
from torchvision import transforms
import cv2
import json
import numpy as np 

json_file = open('../data/way_splits/valUnseen_data.json')
json_data = json.load(json_file)

index = 4

floor = json_data[index]['finalLocation']["floor"]
x, y = json_data[index]['finalLocation']["pixel_coord"]
scan_name = json_data[index]['scanName']

def collect_xs(json_data):
    xs = []
    ind = []
    for i in range(len(json_data)):
        x, y = json_data[i]['finalLocation']["pixel_coord"]
        xs.append(x)
        ind.append(i)
        

    return xs, ind

a, b = collect_xs(json_data)

print(len([_ for _ in a if _ < 200]), len(json_data))

image = Image.open(f'../data/floorplans/floor_{floor}/{scan_name}_{floor}.png')

def convert_image_to_rgb(image):
            return image.convert("RGB")
print(image.size)
preprocess = transforms.Compose([
            transforms.CenterCrop((700, 800)),
            # transforms.Resize(size=(448, 448), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            convert_image_to_rgb,
            # transforms.ToTensor(),
            # transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])


# translate x 

x = x - 200

x = x*(448/800)
y = y*(448/700)
print(x, y)
im2 = preprocess(image)
draw = ImageDraw.Draw(im2)
print('im2size', im2.size)
draw.ellipse((x-10, y-10, x+10, y+10), 'red')


im2.show()