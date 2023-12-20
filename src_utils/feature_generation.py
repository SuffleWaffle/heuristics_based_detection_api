from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from src_utils.colors_processing import decimal_to_rgb, hex_to_rgb


def get_features_mode(features, substitute_val=0):
    counter = Counter(features)
    max_occurencies = max(counter.values())
    if Counter(list(counter.values()))[max_occurencies]>1:
        if substitute_val=='mean':
            return np.mean(features)
        else:
            return substitute_val
    else:
        return max(counter.items(), key=lambda x: x[1])[0]


def generate_features(grids_with_text, width, height):
    area_overall = width * height
    features = []
    # get all features
    for i in grids_with_text:
        bbox = i['bbox']
        text_inside = i['text']
        bbox_area = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
        bbox_relative_position = (bbox[2] + bbox[0]) // 2, (bbox[1] + bbox[3]) // 2
        bbox_relative_position = bbox_relative_position[0] / width, bbox_relative_position[1] / height

        text_bbox = text_inside['x0'], text_inside['y0'], text_inside['x1'], text_inside['y1']

        text_relative_position = (text_bbox[2] + text_bbox[0]) // 2, (text_bbox[1] + text_bbox[3]) // 2
        text_relative_position = text_relative_position[0] / width, text_relative_position[1] / height

        text_bbox_area = abs(text_bbox[2] - text_bbox[0]) * abs(text_bbox[3] - text_bbox[1])

        features.append(
            {'bbox': bbox, 'size': text_inside['spans'][0]['size'], 'message': text_inside['spans'][0]['message'],
             'text_color': decimal_to_rgb(text_inside['spans'][0]['color']), 'text_bbox_area': text_bbox_area,
             'obj_bbox_area': bbox_area, 'obj_text_ratio': 100 * text_bbox_area / bbox_area,
             'text_relative_position_x': text_relative_position[0],
             'text_relative_position_y': text_relative_position[1],
             'bbox_relative_position_x': bbox_relative_position[0],
             'bbox_relative_position_y': bbox_relative_position[1],
             'num_objects': len(grids_with_text),
             'grid_color': hex_to_rgb(i['color']),
             'overall_area': area_overall})
    # joint features
    joint_features = {
        'obj_bbox_area_min': min([j['obj_bbox_area'] \
                                  for j in features]),
        'obj_bbox_area_max': max([j['obj_bbox_area'] \
                                  for j in features])
    }
    # additional features
    for j in features:
        additional_features = {
            'obj_bbox_min_diff': abs(j['obj_bbox_area'] - joint_features['obj_bbox_area_min']),
            'obj_bbox_max_diff': abs(j['obj_bbox_area'] - joint_features['obj_bbox_area_max']),
            'obj_bbox_min_ratio': joint_features['obj_bbox_area_min'] / j['obj_bbox_area'],
            'obj_bbox_max_ratio': j['obj_bbox_area'] / joint_features['obj_bbox_area_max']
        }

        j.update(joint_features)
        j.update(additional_features)

    for idx1, j in enumerate(features):
        bbox_relative_position_x_min_dist = [abs(j['bbox_relative_position_x'] - z['bbox_relative_position_x']) \
                                             for idx2, z in enumerate(features) if idx1 != idx2]

        if bbox_relative_position_x_min_dist:
            bbox_relative_position_x_min_dist = min(bbox_relative_position_x_min_dist)
        else:
            bbox_relative_position_x_min_dist = 0

        bbox_relative_position_y_min_dist = [abs(j['bbox_relative_position_y'] - z['bbox_relative_position_y']) \
                                             for idx2, z in enumerate(features) if idx1 != idx2]

        if bbox_relative_position_y_min_dist:
            bbox_relative_position_y_min_dist = min(bbox_relative_position_y_min_dist)
        else:
            bbox_relative_position_y_min_dist = 0

        on_same_line_x_percentage = np.mean(
            [int(abs(j['bbox_relative_position_x'] - z['bbox_relative_position_x']) < 0.5) for idx2, z in
             enumerate(features) if idx1 != idx2])
        on_same_line_y_percentage = np.mean(
            [int(abs(j['bbox_relative_position_y'] - z['bbox_relative_position_y']) < 0.5) for idx2, z in
             enumerate(features) if idx1 != idx2])

        j.update({'bbox_relative_position_x_min_dist': bbox_relative_position_x_min_dist,
                  'bbox_relative_position_y_min_dist': bbox_relative_position_y_min_dist,
                  'on_same_line_x_percentage': on_same_line_x_percentage,
                  'on_same_line_y_percentage': on_same_line_x_percentage
                  })

    for idx1, j in enumerate(features):
        obj_area = j['obj_bbox_area']
        area_eq_perc = np.mean(
            [int(abs(1 - obj_area / z['obj_bbox_area']) < 0.1) for idx2, z in enumerate(features) if idx1 != idx2])

        actual_position = [j['bbox_relative_position_x'], j['bbox_relative_position_y']]
        positions = [[z['bbox_relative_position_x'], z['bbox_relative_position_y']] \
                     for idx2, z in enumerate(features) if idx1 != idx2]

        distances = euclidean_distances([actual_position], positions)
        j.update({'mean_distance_to_objects': distances.mean(),
                  'mean_distance_to_3_closest': np.mean(np.sort(distances[0])[:3]),
                  'area_eq_perc': area_eq_perc})

    features_df = pd.DataFrame(features)
    features_df['text_color_red'] = features_df['text_color'].apply(lambda x: x[0]) / 255
    features_df['text_color_green'] = features_df['text_color'].apply(lambda x: x[1]) / 255
    features_df['text_color_blue'] = features_df['text_color'].apply(lambda x: x[2]) / 255
    features_df['grid_color_red'] = features_df['grid_color'].apply(lambda x: x[0]) / 255
    features_df['grid_color_green'] = features_df['grid_color'].apply(lambda x: x[1]) / 255
    features_df['grid_color_blue'] = features_df['grid_color'].apply(lambda x: x[2]) / 255
    features_df['obj_bbox_min_diff'] = features_df['obj_bbox_min_diff'] / features_df['overall_area'] * 100
    features_df['obj_bbox_max_diff'] = features_df['obj_bbox_max_diff'] / features_df['overall_area'] * 100
    features_df['obj_bbox_area'] = features_df['obj_bbox_area'] / features_df['overall_area'] * 100
    return features_df