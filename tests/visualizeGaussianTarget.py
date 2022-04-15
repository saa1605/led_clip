from pydoc import tempfilepager
import cv2 
import numpy as np 
import json 
import torch 
from scipy.ndimage.filters import gaussian_filter

max_floors = 5
json_file = open('../data/way_splits/train_data.json')
json_data = json.load(json_file)

mesh_file = open('../data/floorplans/pix2meshDistance.json')
mesh2meters = json.load(mesh_file)

index = 1

floor = json_data[index]['finalLocation']["floor"]
x, y = json_data[index]['finalLocation']["pixel_coord"]
scan_name = json_data[index]['scanName']
mesh_conversion = mesh2meters[scan_name][str(floor)]["threeMeterRadius"] / 3.0
temperature = 0.02
def create_target(level, location, mesh_conversion, temperature):
        gaussian_target = np.zeros(
            (max_floors, int(700*0.65), int(1200*0.65))
        )
        gaussian_target[level, location[0], location[1]] = 1./temperature
        print(gaussian_target[level, location[0], location[1]] )
        gaussian_target[level, :, :] = gaussian_filter(
            gaussian_target[int(level), :, :],
            sigma=(mesh_conversion/60),
        )
        gaussian_target[int(level), :, :] = (
            gaussian_target[int(level), :, :]
            / gaussian_target[int(level), :, :].sum()
        )
        # gaussian_target = torch.tensor(gaussian_target)
        return gaussian_target



target = create_target(floor, [y, x], mesh_conversion, temperature)

print(np.unique(target[floor]))
print(target[floor].max())

# cv2.imshow('target', target[floor])
# cv2.waitKey()

