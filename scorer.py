'''
scorer.py provded by the competition hosts
'''

import argparse
import numpy as np
import json
import os
import traceback
from shapely.ops import unary_union
from shapely.geometry import Polygon
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

SIZE                            = 512
MAX_BUILDINGS                   = 1000
MAX_VERTICES                    = 300
MAX_VERTICES_TOTAL              = 5000
MAX_HEIGHT                      = 1000
BUILDING_CORRECT_AREA_THRESHOLD = 0.5
BUILDINGS_IOU_THRESHOLD         = 0.1
SHOW_PROGRESS_BAR               = True
PRINT_DETAILED_SCORE            = True


class Score:
    def __init__(self):
        self.precision_correct_area = 0
        self.precision_incorrect_area = 0

        self.recall_correct_area = 0
        self.recall_incorrect_area = 0

        self.gt_out_heights = []

    def accumulate(self, other):
        self.precision_correct_area += other.precision_correct_area
        self.precision_incorrect_area += other.precision_incorrect_area
        
        self.recall_correct_area += other.recall_correct_area
        self.recall_incorrect_area += other.recall_incorrect_area

        self.gt_out_heights += other.gt_out_heights

    def get_score(self, *, print_detailed_score=True):
        if self.precision_correct_area + self.precision_correct_area == 0:
            precision = 0
        else:
            precision = self.precision_correct_area / (self.precision_correct_area + self.precision_incorrect_area) * 100
        recall = self.recall_correct_area / (self.recall_correct_area + self.recall_incorrect_area) * 100

        ground_truths = np.array(self.gt_out_heights)[:, 0]
        predictions = np.array(self.gt_out_heights)[:, 1]
        rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))

        score = max(0, int((precision + recall - 4 * rmse) * 50000))
        
        if print_detailed_score:
            print('Building contour precision', precision)
            print('Building contour recall', recall)
            print("RMSE of building heights: ", rmse)
            print("Final Score: ", score)

        return precision, recall, rmse, score
    

class Building:
    def __init__(self, polygon, height):
        self.polygon = polygon
        self.height = height


def get_all_files(dir):
    all_files = {}
    for path, _, files in os.walk(dir):
        for file in files:
            if file in all_files:
                raise Exception(f"multiple occurences of the same file: {file}")
            all_files[file] = os.path.join(path, file)
    return all_files


def read_buildings(file_path):
    buildings = []
    with open(file_path) as json_file:
        data = json.load(json_file)
        if 'shapes' not in data:
            raise Exception(f"invalid file format (no 'shapes' field): {file_path}")
        shapes = data['shapes']
        total_vertices = 0
        for s in shapes:
            if 'points' not in s:
                raise Exception(f"invalid file format (no 'points' field): {file_path}")
            if len(s['points']) > MAX_VERTICES:
                raise Exception(f"too much points for polygon: {file_path}")
            total_vertices += len(s['points'])
            if total_vertices > MAX_VERTICES_TOTAL:
                raise Exception(f"total points limit exceeded: {file_path}")
            polygon = Polygon(s['points'])
            if not polygon.is_simple:
                raise Exception(f"invalid polygon: {file_path}")
            if 'group_id' not in s:
                raise Exception(f"invalid file format (no 'group_id' field): {file_path}")
            height = s['group_id']

            buildings += [Building(polygon, height)]

    return buildings


def check_bounding_box_intersection(building_a, building_b):
    a_min_x, a_min_y, a_max_x, a_max_y = building_a.polygon.bounds
    b_min_x, b_min_y, b_max_x, b_max_y = building_b.polygon.bounds
    if a_max_x < b_min_x or a_min_x > b_max_x or a_max_y < b_min_y or a_min_y > b_max_y:
        return False
    else:
        return True
    

def read_out_buildings(file_path, file_name):
    buildings = read_buildings(file_path)

    for building in buildings:
        minx, miny, maxx, maxy = building.polygon.bounds
        if minx < 0 or miny < 0 or maxx > SIZE or maxy > SIZE:
            raise Exception(f'point out of bounds in file {file_name}')

        if building.height < 0 or building.height > MAX_HEIGHT:
            raise Exception(f'building height out of range in file {file_name}')

    for i in range(len(buildings)):
        for j in range(i+1, len(buildings)):
            if (check_bounding_box_intersection(buildings[i], buildings[j])):
                intersection = buildings[i].polygon.intersection(buildings[j].polygon).area
                if intersection / min(buildings[i].polygon.area, buildings[j].polygon.area) > BUILDINGS_IOU_THRESHOLD:
                    raise Exception(f'two polygons in file {file_name} have iou larger than the threshold')
            
    return buildings

def get_out_buildings(out_files, file_name):
    if not file_name in out_files:
        raise Exception(f"file {file_name} not found in output directory")
    
    return read_out_buildings(out_files[file_name], file_name)


def get_score_for_absent_file(gt_buildings):
    score = Score()
    for b in gt_buildings:
        score.recall_incorrect_area += b.polygon.area
        score.gt_out_heights += [(b.height, 0)]

    return score


def get_score_for_one_file_buildings(gt_buildings, out_buildings):
    gt_union = unary_union([b.polygon for b in gt_buildings])
    out_union = unary_union([b.polygon for b in out_buildings])

    score = Score()

    for out_building in out_buildings:
        intersection_area = out_building.polygon.intersection(gt_union).area
        if intersection_area > out_building.polygon.area * BUILDING_CORRECT_AREA_THRESHOLD:
            score.precision_correct_area += out_building.polygon.area
        else:
            score.precision_incorrect_area += out_building.polygon.area
    
    for gt_building in gt_buildings:
        intersection_area = gt_building.polygon.intersection(out_union).area
        if intersection_area > gt_building.polygon.area * BUILDING_CORRECT_AREA_THRESHOLD:
            score.recall_correct_area += gt_building.polygon.area
        else:
            score.recall_incorrect_area += gt_building.polygon.area

    out_buildings_found = [False] * len(out_buildings)
    gt_buildings_found = [False] * len(gt_buildings)

    for i, out_building in enumerate(out_buildings):
        for j, gt_building in enumerate(gt_buildings):
            if (check_bounding_box_intersection(out_building, gt_building)):
                intersection_area = out_building.polygon.intersection(gt_building.polygon).area
                if intersection_area >= min(out_building.polygon.area, gt_building.polygon.area) * BUILDING_CORRECT_AREA_THRESHOLD:
                    score.gt_out_heights += [(gt_building.height, out_building.height)]
                    out_buildings_found[i] = True
                    gt_buildings_found[j] = True

    for i, out_building in enumerate(out_buildings):
        if not out_buildings_found[i]:
            score.gt_out_heights += [(0, out_building.height)]

    for j, gt_building in enumerate(gt_buildings):
        if not gt_buildings_found[j]:
            score.gt_out_heights += [(gt_building.height, 0)]

    return score


def get_score_for_one_file(gt_files, out_files, file):
    gt_buildings = read_buildings(gt_files[file])
    try:
        out_buildings = get_out_buildings(out_files, file)
    except:
        return get_score_for_absent_file(gt_buildings)
    return get_score_for_one_file_buildings(gt_buildings, out_buildings)



def get_score(gt_dir, out_dir):
    if not os.path.exists(out_dir):
        raise Exception("output directory doesn't exist")
    if not os.path.exists(gt_dir):
        raise Exception("ground truth directory doesn't exist")
    
    gt_files = get_all_files(gt_dir)
    out_files = get_all_files(out_dir)

    score = Score()
    
    for file in tqdm(gt_files.keys(), disable=not SHOW_PROGRESS_BAR):
        file_score = get_score_for_one_file(gt_files, out_files, file)
        score.accumulate(file_score)

    return score


def main(gt_dir, out_dir):
    try:
        score = get_score(gt_dir, out_dir)
        print(score.get_score(print_detailed_score=PRINT_DETAILED_SCORE))
    except:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--ground-truth-dir', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    args = parser.parse_args()
    main(args.ground_truth_dir, args.output_dir)




