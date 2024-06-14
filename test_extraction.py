import json
import numpy as np
from PIL import Image, ImageDraw
import os
import cv2

def create_binary_mask(label_path, output_mask_path, width=512, height=512):
    # Load the JSON file
    with open(label_path, 'r') as f:
        data = json.load(f)

    # Create a blank binary mask
    mask = Image.new('1', (width, height), 0)

    # Draw polygons on the mask
    for shape in data['shapes']:
        polygon = np.array(shape['points']).flatten().tolist()
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)

    # Save the mask
    mask.save(output_mask_path)

    mask_np = np.asarray(mask)
    print((mask_np == 1).sum(), 512*512)

def mask_to_labelme(mask, output_json_path):
    # Load the binary mask
    # mask = np.array(Image.open(mask_path).convert('L'))

    # Find contours in the mask
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create LabelMe JSON structure
    labelme_data = {
        "shapes": [],
    }
    print(len(contours), mask.sum(), mask.shape, 512*512)
    for contour in contours:
        print(contour)
        contour = contour.squeeze(axis=1)  # Remove unnecessary dimension
        points = contour.tolist()

        shape = {
            "label": 'building',
            "points": points,
            "group_id": 0,
            "shape_type": "polygon",
            "flags": {}
        }
        labelme_data["shapes"].append(shape)

    # Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)

# Example usage

output_mask_path = 'temp.png'
create_binary_mask('temp_train_results/aabijjmwby.json', output_mask_path)


mask = np.asarray(Image.open(output_mask_path).convert('L')) // 255
print(mask.dtype, mask.shape)
print((mask == 1).sum(), 512*512)

output_json_path = 'temp.json'
mask_to_labelme(mask, output_json_path)

output_rec_mask = 'temp_rec.png'
create_binary_mask(output_json_path, output_rec_mask)
