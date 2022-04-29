import cv2
import json

json_file = open('../data/way_splits/train_data.json')
json_data = json.load(json_file)

index = 6

floor = json_data[index]['finalLocation']["floor"]
x, y = json_data[index]['finalLocation']["pixel_coord"]
scan_name = json_data[index]['scanName']

image = cv2.imread(f'../data/floorplans/floor_{floor}/{scan_name}_{floor}.png')

# scale_percent = 65  # percent of original size
x = int(x * (448/image.shape[1]))
y = int(y * (448/image.shape[0]))
# resize image
image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_CUBIC)

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

# image = center_crop(image, (448, 448))

print(x, y)

image = cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)


cv2.imshow('imageWithLocation', image)
cv2.waitKey()
