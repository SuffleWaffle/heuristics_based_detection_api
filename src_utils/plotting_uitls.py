import cv2
from copy import deepcopy
import PIL.Image as pil_image
import numpy as np

def plot_detection_result(img_array, res_dict):
    test_img = deepcopy(np.array(img_array))
    for obj in res_dict['data']:
        x_min, y_min, x_max, y_max = list(obj['bbox'].values())
        cv2.rectangle(test_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return pil_image.fromarray(test_img)

def plot_grids_result(img_array, grid_lines):
    test_img = deepcopy(np.array(img_array))
    for i in grid_lines:
        cv2.rectangle(test_img, i['bbox'][:2],
                      i['bbox'][2:],
                      color=(255,0,0),
                      thickness=2)
        cv2.line(test_img, i['grid'][:2],
                 i['grid'][:2],
                 color=(0,255,0), thickness=2)
    return pil_image.fromarray(test_img)

def plot_jb_ds_result(img_array, classified):
    test_img = deepcopy(np.array(img_array))
    for i in classified:
        cv2.rectangle(test_img, list(i['bbox'].values())[:2], list(i['bbox'].values())[2:],
                      color=(255, 0, 0))
    return pil_image.fromarray(test_img)

def plot_closed_objects_sld(lines, closed_objects):
    test_img = np.full(shape=(int(closed_objects['height']),
                              int(closed_objects['width']), 3), fill_value=255,
                       dtype=np.uint8)

    for line in lines['lines_data']:
        cv2.line(test_img, line[:2], line[2:], color=(0, 0, 0), thickness=1)
    pil_image.fromarray(test_img)

    for i in closed_objects['data']:
        x0, y0, x1, y1 = list(i['bbox'].values())
        cv2.rectangle(test_img, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)
    return pil_image.fromarray(test_img)


