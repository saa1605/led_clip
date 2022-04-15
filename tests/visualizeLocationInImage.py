import cv2
import json

json_file = open('../data/way_splits/train_data.json')
json_data = json.load(json_file)

index = 6

floor = json_data[index]['finalLocation']["floor"]
x, y = json_data[index]['finalLocation']["pixel_coord"]
scan_name = json_data[index]['scanName']

image = cv2.imread(f'../data/floorplans/floor_{floor}/{scan_name}_{floor}.png')

scale_percent = 65  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
print(dim)
# resize image
# image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# x = int(x * 0.65)
# y = int(y * 0.65)

image = cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)


cv2.imshow('imageWithLocation', image)
cv2.waitKey()
