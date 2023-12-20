from itertools import chain

from fastapi.exceptions import HTTPException

from src_utils.geometry_utils import fix_coords_line, check_line_type
from src_utils.graph_utils import find_cyclic_objects, merge_small_lines_all
from src_utils.heuristics_utils import get_bbox_circle_objects, get_bbox_rectangles_objects, get_closed_objects, \
    filter_by_area_bigger, get_objects_with_no_insides, filter_objects_by_proportion, get_text_inside_objects, \
    get_text_closest_to_center, associate_text_with_info, find_possible_switches, classify_by_exact_text_inside_object, \
    find_disconnected_switches, filter_disconnected_switches, form_bbox_switch, \
    classify_by_exact_text, merge_bboxes, is_parallelogram, delete_objects_inside_object
from src_utils.lines_processing import filter_by_color_in
from src_logging.log_config import setup_logger
import logging
logger = setup_logger(logger_name=__name__, logging_level=logging.INFO)


def detect_by_text_n_disconnected_heuristic(lines: dict,
                                            cubic_lines: dict,
                                            circles: dict,
                                            rectangles: dict,
                                            quads: dict,
                                            parsed_text: dict,
                                            config: dict,
                                            to_return_lines=False):
    # unravel lines
    try:
        lines_attributes = lines['lines_attributes']
        logger.info('- LINES - Unravelled svg elements attributes')

        svg_width, svg_height = lines['svg_width'], lines['svg_height']
        logger.info('- LINES - Unravelled svg width and height')

        lines = lines['lines_data']
        logger.info('- LINES - Unravelled svg elements data')
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with lines data : {str(e)}')

    # unravel cubic lines
    try:
        cub_bezier_lines, cub_attributes = cubic_lines['cub_bezier_lines_data'],\
            cubic_lines['attributes']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with cubic data : {str(e)}')

    # unravel circles
    try:
        circles, circle_attributes = circles['circles_data'],\
            circles['attributes']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with circles data : {str(e)}')

    # unravel rectangles
    try:
        rectangles, rectangles_attributes = rectangles['rectangles_data'],\
            rectangles['attributes']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with rectangles data : {str(e)}')

    # unravel quads
    try:
        quads_objects, quad_attributes = (quads['quad_bezier_lines_data'],
                                          quads['attributes'])
        logger.info('- QUADS - Unravelled svg elements data')
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with quads data : {str(e)}')

    # unravel text
    try:
        parsed_text = parsed_text['parsed_text']
        logger.info('- PARSED TEXT - Unravelled text')
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with parsed text data : {str(e)}')


    logger.info('Unravelled svg elements data and text')

    # get everything togather and filter by color
    lines += cub_bezier_lines
    lines_attributes += cub_attributes
    lines = filter_by_color_in(lines, lines_attributes,
                               **config['filter_by_color_in'])
    circles = filter_by_color_in(circles, circle_attributes,
                                 **config['filter_by_color_in'])
    rectangles = filter_by_color_in(rectangles, rectangles_attributes,
                                    **config['filter_by_color_in'])
    quads_objects = filter_by_color_in(quads_objects, quad_attributes,
                                       **config['filter_by_color_in'])

    # normalize liens
    lines = [fix_coords_line(line) for line in lines]
    lines = list(set(lines))
    lines = [i for i in lines if not (i[0]==i[2] and i[1]==i[3])]
    logger.info('Normalized lines')

    # find cyclic objects
    cyclic_objects = find_cyclic_objects(lines)
    closed_objects = get_closed_objects(cyclic_objects, lines, svg_width, svg_height,
                                        **config['get_closed_objects'],
                                        filter_func=filter_by_area_bigger)
    # get circle objects
    circles_objects = get_bbox_circle_objects(circles)

    # get rectangles objects
    rectangles_objects = get_bbox_rectangles_objects(rectangles)

    # get everything togather
    closed_objects += rectangles_objects + circles_objects + quads_objects
    closed_objects = merge_bboxes(closed_objects)

    logger.info('Got closed objects')

    # get objects with no objects inside
    closed_objects_no_insides = get_objects_with_no_insides(closed_objects)
    closed_objects_filtered = filter_objects_by_proportion(closed_objects_no_insides,
                                                           **config['filter_objects_by_proportion'])

    # get text inside objects
    text_inside_objects =  get_text_inside_objects(closed_objects_filtered, parsed_text,
    **config['get_text_inside_objects'])
    # get text closer to center of bbox
    text_inside_objects = get_text_closest_to_center(text_inside_objects)
    # associate text with info
    text_inside_objects = associate_text_with_info(text_inside_objects, parsed_text)
    # find possible disconnected switches
    possible_switches = find_possible_switches(text_inside_objects,
                           closed_objects_filtered)
    # limit text to only one entry
    text_inside_objects_one_entry = []
    text_multiple_entries = []
    for k, v in text_inside_objects.items():
        if len(v) == 1 and len(v[0]['spans']) == 1:
            text_inside_objects_one_entry.append([v[0], k])
        else:
            text_multiple_entries.extend(v)

    logger.info('Filtered objects, found possible switches')

    classified = classify_by_exact_text_inside_object(text_inside_objects_one_entry,
                                                      **config['classify_by_exact_text_inside_object'])
    logger.info('Classified by text heuristic inside closed objects')

    # classify by text only
    text_inside_objects_one_entry_matched = [i[0] for i in text_inside_objects_one_entry]
    to_check_text = [i for i in parsed_text if i not in text_inside_objects_one_entry_matched]
    to_check_text = [i for i in to_check_text if i not in text_multiple_entries]
    classified.extend(classify_by_exact_text(to_check_text,
                                             **config['classify_by_exact_text_only']))
    logger.info('Classified by text heuristic only')

    # find closed objects
    classified_coords = set([tuple(i['bbox'].values()) for i in classified])
    closed_objects = [i for i in closed_objects if tuple(i['bbox'].values()) not in classified_coords]

    # filter lines
    rectangles = []
    for i in possible_switches:
        try:
            if is_parallelogram(merge_small_lines_all(i['lines'])):
                rectangles.append(i)
        except:
            pass

    lines_of_objects = list(map(fix_coords_line,\
                                list(chain(*[i['lines'] for i in closed_objects_no_insides]))))
    lines = [i for i in lines if not (i[0] == i[2] and i[1] == i[3])]
    lines_of_objects = [i for i in lines_of_objects if not (i[0] == i[2] and i[1] == i[3])]
    lines = list(set(lines).difference(lines_of_objects))
    lines_to_check = [line for line in lines if check_line_type(line) \
                      in ['horizontal', 'vertical']]

    # find disconnected switches
    if config['detect_switches']:
        disconnected_switches = find_disconnected_switches(rectangles,
                                                           lines_to_check, lines)
        # filter disconnected switches
        disconnected_switches = filter_disconnected_switches(disconnected_switches,
                                                             closed_objects_no_insides,
                                                             lines_to_check,
                                                             **config['filter_disconnected_switches'])
        # form bbox
        closed_objects, disconnected_switches = \
            form_bbox_switch(candidates=closed_objects, disconnected_switches=disconnected_switches,
                             to_return_lines=to_return_lines)
        logger.info('Detected disconnected switches')

        classified = delete_objects_inside_object(disconnected_switches,\
                                                             classified)
        # get everything togather
        classified = disconnected_switches + classified
    for i in classified:
        i['confidence'] = 1.0
    classified = {'data': classified,
                  'width' : svg_width,
                  'height' : svg_height}

    return classified
