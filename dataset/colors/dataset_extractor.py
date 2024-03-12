import os
import cv2
import json

folder_path = '/home/ben/Code/Flow-Free-Solver/dataset/colors'

output_file = 'dataset.json'


data = {}

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):

        name, ext = os.path.splitext(filename)
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        center_pixel_color = image[center_y, center_x]

        data[int(name)] = center_pixel_color.tolist()
        

sorted_keys = sorted(data.keys())
sorted_data = {}

for key in sorted_keys:
    sorted_data[key] = data[key]

with open(output_file, 'w') as f:
    json.dump(sorted_data, f, indent=4)

print("Dataset created and saved successfully.")