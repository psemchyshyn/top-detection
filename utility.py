import numpy as np
import cv2
import json
import shapely
from shapely.geometry import Polygon
from PIL import Image, ImageDraw


def labelme2mask(labelme_json, image_w, image_h):
    mask_roof = Image.new('1', (image_w, image_h), 0)
    mask_height = np.zeros((image_w, image_h))

    for shape in labelme_json['shapes']:
        polygon = np.array(shape['points']).flatten().tolist()
        temp_mask = Image.new('1', (image_w, image_h), 0)
        ImageDraw.Draw(mask_roof).polygon(polygon, outline=1, fill=1)
        ImageDraw.Draw(temp_mask).polygon(polygon, outline=1, fill=1)

        height = shape['group_id']
        mask_height[np.asarray(temp_mask) == 1] = height

    return np.uint8(np.asarray(mask_roof)), mask_height


def get_height_of_the_contour(mask_height, contour):
    contour_mask = Image.new('1', mask_height.shape, 0)
    contour = np.array(contour).flatten().tolist()
    ImageDraw.Draw(contour_mask).polygon(contour, outline=1, fill=1)
    contour_mask = np.asarray(contour_mask)
    height = mask_height[contour_mask == 1].mean()

    if height < 0:
        height = 0
    elif height > 5000:
        height = 0
    else:
        height = height
    return float(height)


def mask2labelme(mask, mask_height, output_json_path, mask_threshold=0.5, max_vertices=8, epsilon=0.01): # mask is np.array with 0 and 1 values
    mask = np.uint8(mask > mask_threshold)
    # print(mask.shape, mask.dtype, mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labelme_data = {
        "shapes": [],
    }
    # print('stat')
    # print(len(contours), mask.sum(), mask.shape, 512*512)
    for contour in contours:
        if max_vertices is not None:
            epsilon_val = epsilon * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon_val, True)

            while len(approx_contour) > max_vertices and epsilon_val < 1:
                    epsilon_val *= 1.1
                    approx_contour = cv2.approxPolyDP(contour, epsilon_val, True)

        approx_contour = approx_contour.squeeze(axis=1).tolist()  # Remove unnecessary dimension

        if len(approx_contour) <= 2:
            continue

        polygon = Polygon(approx_contour)
        polygon = shapely.simplify(polygon, tolerance=4, preserve_topology=True)

        if not polygon.is_simple:
            continue

        points = list(polygon.exterior.coords)[:-1]


        height = get_height_of_the_contour(mask_height, points)

        shape = {
            "points": points,
            "group_id": height,
        }

        labelme_data["shapes"].append(shape)

    # Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)
