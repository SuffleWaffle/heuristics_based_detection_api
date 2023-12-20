from src_logging.log_config import setup_logger
from src_utils.geometry_utils import fix_coords_line, split_all_lines_by_intersections
from src_utils.graph_utils import create_graph_points, find_cyclic_objects
from src_utils.heuristics_utils import filter_cyclic_objects, get_closed_objects, get_bbox_circle_objects, \
    get_bbox_rectangles_objects, merge_bboxes_sld, filter_by_area_bigger, filter_closed_objects
from src_utils.lines_processing import filter_by_color_nin, del_by_existence_lines, del_by_existence_objects
import numpy as np
logger = setup_logger(__name__)
def find_closed_objects_sld(lines:dict,
                            cubic_lines:dict,
                            rectangles:dict,
                            circles:dict,
                            quads:dict,
                            img_processed:np.array,
                            config:dict):
    # unravell lines
    svg_height, svg_width = lines['svg_height'], lines['svg_width']
    lines, lines_attributes = lines['lines_data'], \
        lines['lines_attributes']
    cubic_lines, cubic_attributes = cubic_lines['cub_bezier_lines_data'], \
        cubic_lines['attributes']
    lines += cubic_lines
    lines_attributes += cubic_attributes
    lines = list(map(fix_coords_line, lines))
    dict_lines_attributes = {}
    for c, i in enumerate(lines):
        dict_lines_attributes[i] = lines_attributes[c]['color']

    circles, circles_attributes = circles['circles_data'], \
        circles['attributes']
    quads, quads_attributes = quads['quad_bezier_lines_data'], \
        quads['attributes']
    rectangles, rectangles_attributes = rectangles['rectangles_data'], \
        rectangles['attributes']

    logger.info('Unravelled svg objects')

    lines = filter_by_color_nin(lines, lines_attributes, **config['filter_by_color_nin'])
    circles = filter_by_color_nin(circles, circles_attributes, **config['filter_by_color_nin'])
    quads = filter_by_color_nin(quads, quads_attributes, **config['filter_by_color_nin'])
    rectangles = filter_by_color_nin(rectangles, rectangles_attributes, **config['filter_by_color_nin'])

    logger.info('Filtered by color')

    to_del = del_by_existence_lines(img_processed=img_processed, lines=lines,
                                    **config['del_by_existence_lines'])
    lines = list(set(lines).difference(to_del))
    lines = [i for i in lines if not (i[0] == i[2] and i[1] == i[3])]

    logger.info('Deleted lines by existence on image')
    logger.info(f'Number of lines before : {len(lines)}')
    original_lines, lines = split_all_lines_by_intersections(lines,
                                                             **config['split_all_lines_by_intersections'])

    logger.info(f'Split lines by intersections, number of new lines : {len(lines)}')

    possible_cycles = find_cyclic_objects(lines)
    filtered_cycles = filter_cyclic_objects(possible_cycles)
    logger.info(f'Filtered cycles number : {len(filtered_cycles)}')
    logger.info('Found possible cycles')

    closed_objects = get_closed_objects(filtered_cycles, lines,
                       svg_width=svg_width,
                       svg_height=svg_height,
                                        filter_func=filter_by_area_bigger,
                                        **config['get_closed_objects'])
    logger.info(f'Number of closed objects found by graph : {len(closed_objects)}')
    circles_objects = get_bbox_circle_objects(circles)
    rectangles_objects = get_bbox_rectangles_objects(rectangles)

    logger.info('Got closed objects')

    closed_objects = merge_bboxes_sld(closed_objects)

    logger.info('Merged bboxes for closed objects')

    closed_objects += del_by_existence_objects(img_processed, rectangles_objects,
                                               **config['del_by_existence_objects']) \
                      + del_by_existence_objects(img_processed, circles_objects,
                                                 **config['del_by_existence_objects']) \
                      + del_by_existence_objects(img_processed, quads,
                                                 **config['del_by_existence_objects'])
    closed_objects = filter_closed_objects(closed_objects)

    logger.info('Enriched closed objects with other objects from svg')

    closed_objects = {'data': closed_objects,
                 'width' : svg_width,
                 'height' : svg_height}
    logger.info(f'Number of final objects: {len(closed_objects["data"])}')

    lines_attributes = []
    for i in original_lines:
        lines_attributes.append(dict_lines_attributes[tuple(i)])
    lines = {'lines_data': original_lines,
             'attributes': lines_attributes}

    return closed_objects, lines



