import json 
import torch 
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter
import numpy as np 
import src.clip as clip 

class LEDDataset(Dataset):
    def __init__(self, data_path, image_dir, config):

        # Gather train_data from {train/val/test}_data.json
        self.data_path = data_path 
        self.data_file = open(self.data_path)
        self.data = json.load(self.data_file)

        # Extract the mode (train, valSeen, valUnseen) from the data_path 
        self.mode = self.data_path.split('/')[-1][:-5].split('_')[0]

        # Store access to floorplans directory 
        self.image_dir = image_dir 

        # Save the global config 
        self.config = config 

        # Calculate parameters to adjust location based on scaling and cropping
        self.crop_translate_x = (self.config['original_image_size'][2] - self.config['cropped_image_size'][2])/2 
        self.crop_translate_y = (self.config['original_image_size'][1] - self.config['cropped_image_size'][1])/2 

        self.resize_scale_x = self.config['scaled_image_size'][2] / self.config['cropped_image_size'][2]
        self.resize_scale_y = self.config['scaled_image_size'][1] / self.config['cropped_image_size'][1]

        # mesh2meters
        self.mesh2meters_path = self.config['mesh2meters']
        self.mesh2meters_file = open(self.mesh2meters_path)
        self.mesh2meters = json.load(self.mesh2meters_file)

        # transform required for CLIP 
        def convert_image_to_rgb(image):
            return image.convert("RGB")

        self.preprocess = transforms.Compose([
            transforms.CenterCrop((self.config['cropped_image_size'][1], self.config['cropped_image_size'][2])),
            transforms.Resize(size=(self.config['scaled_image_size'][1], self.config['scaled_image_size'][2]), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess_visualize = transforms.Compose([
            transforms.CenterCrop((self.config['cropped_image_size'][1], self.config['cropped_image_size'][2])),
            transforms.Resize(size=(self.config['scaled_image_size'][1], self.config['scaled_image_size'][2]), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            convert_image_to_rgb,
        ]
        )


    def gather_all_floors(self, index):
        ''' Collect images for all floor for a given index
            Ouptut: Maps(max_floors, (image_size)), Conversions(max_floors, 1)
        '''

        # Create empty tensors to hold maps and conversions 
        all_maps = torch.zeros(
            self.config['max_floors'],
            self.config["image_size"][0],
            self.config["image_size"][1],
            self.config["image_size"][2],
        )
        all_conversions = torch.zeros(self.config["max_floors"], 1)

        # Extract scan_names and which floors that scan has from the data 
        scan_name = self.data[index]['scanName']
        floors = self.mesh2meters[scan_name].keys()

        # Iterate through each floor of a scan, open the image, preprocess it and convert it to a tensor 
        for enum, floor in enumerate(floors):
            img = Image.open(f'{self.image_dir}floor_{floor}/{scan_name}_{floor}.png').convert('RGB')
            if "train" in self.mode:
                all_maps[enum, :, :, :] = self.preprocess(img)[:3, :, :]
            else:
                all_maps[enum, :, :, :] = self.preprocess(img)[:3, :, :]
            all_conversions[enum, :] = self.mesh2meters[scan_name][floor]["threeMeterRadius"] / 3.0
        return all_maps, all_conversions

    def gather_correct_floor(self, index):
        scan_name = self.data[index]['scanName']
        x, y, floor = self.scale_location(index)
        img = Image.open(f'{self.image_dir}floor_{floor}/{scan_name}_{floor}.png').convert('RGB')
        
        map = self.preprocess(img)
        conversion = torch.tensor(self.mesh2meters[scan_name][str(floor)]["threeMeterRadius"] / 3.0).float()

        return map, conversion

    def scale_location(self, index):
        if "test" in self.mode:
            return [0, 0, 0]

        floor = self.data[index]['finalLocation']["floor"]
        x, y = self.data[index]['finalLocation']["pixel_coord"]    

        return [int((x - self.crop_translate_x) * self.resize_scale_x), 
                int((y - self.crop_translate_y) * self.resize_scale_y), 
                floor] 
    

    def create_target(self, index, x, y, floor):

        scan_name = self.data[index]['scanName']
        mesh_conversion =(self.mesh2meters[scan_name][str(floor)]["threeMeterRadius"] / 3.0)*(self.config['conversion_scale'])
        print(mesh_conversion)
        gaussian_target = np.zeros(
            (self.config['max_floors'], self.config['image_size'][1], self.config['image_size'][2])
        )
        gaussian_target[floor, y, x] = 1 # y, x because y -> rows and x -> columns
        gaussian_target[floor, :, :] = gaussian_filter(
            gaussian_target[floor, :, :],
            sigma=(mesh_conversion),
        )
        gaussian_target[floor, :, :] = (
            gaussian_target[floor, :, :]
            / gaussian_target[floor, :, :].sum()
        )
        gaussian_target = torch.tensor(gaussian_target)
        return gaussian_target
        
    def join_dialog(self, index):
        dialogArray = self.data[index]['dialogArray']
        return " ".join(dialogArray)
    
    def visualize_target(self, index):
        x, y, floor = self.scale_location(index)
        
        scan_name = self.data[index]['scanName']
        img = Image.open(f'{self.image_dir}floor_{floor}/{scan_name}_{floor}.png').convert('RGB')

        img_vis = self.preprocess_visualize(img)
        draw = ImageDraw.Draw(img_vis)
        draw.ellipse((x-10, y-10, x+10, y+10), 'red')
        print(self.join_dialog(index))
        img_vis.show()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        target_x, target_y, target_floor = self.scale_location(index)
        # maps, conversions = self.gather_correct_floor(index)
        maps, conversions = self.gather_all_floors(index)
        dialogs = clip.tokenize(self.join_dialog(index), truncate=True)
        targets = self.create_target(index, target_x, target_y, target_floor)

        return {
            'maps': maps,
            'target_maps': targets, 
            'conversions': conversions,
            'dialogs': dialogs
        }
            