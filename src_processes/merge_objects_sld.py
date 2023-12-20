import logging

from src_logging.log_config import setup_logger
from src_utils.geometry_utils import is_point_inside_rect, scale
from src_utils.heuristics_utils import pairwise_delete_objects_by_area


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = setup_logger(logger_name=__name__, logging_level=logging.INFO)


def merge_objects_sld(filtered_lines: dict,
                      closed_objects: dict,
                      detected_objects: dict,
                      junction_boxes: dict,
                      config: dict):
    width, height = junction_boxes['width'], junction_boxes['height']
    filtered_lines = filtered_lines['lines_data']
    junction_boxes = junction_boxes['data']
    detected_objects = detected_objects['data']
    closed_objects = closed_objects['data']

    logger.info('Unravelled closed objects, junction boxes and lines')
    for i in junction_boxes:
        i['class_heuristic'] = i['sub_class']
        del i['class']
        del i['sub_class']
    indices_to_del = set(pairwise_delete_objects_by_area(junction_boxes, closed_objects,
                                                         **config['pairwise_delete_objects_by_area']) + \
                         pairwise_delete_objects_by_area(detected_objects, closed_objects,
                                                         **config['pairwise_delete_objects_by_area']))
    closed_objects = [i for c, i in enumerate(closed_objects) if c not in indices_to_del]
    indices_to_del = set(pairwise_delete_objects_by_area(junction_boxes, detected_objects,
                                                         **config['pairwise_delete_objects_by_area']))
    junction_boxes = [i for c, i in enumerate(junction_boxes) if c not in indices_to_del]
    closed_objects += junction_boxes + detected_objects
    logger.info('Merged objects together')
    for ent in closed_objects:
        bbox = list(ent['bbox'].values())
        lines_in_bbox = []
        if not ent.get('lines'):
            for line in filtered_lines:
                if is_point_inside_rect(line[:2], bbox) and is_point_inside_rect(line[2:], bbox):
                    lines_in_bbox.append(line)
            ent['lines'] = lines_in_bbox
    logger.info('Found lines inside objects')
    closed_objects = {'data': closed_objects,
                      'width': width,
                      'height': height}

    return closed_objects
