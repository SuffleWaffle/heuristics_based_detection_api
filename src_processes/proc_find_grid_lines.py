from src_utils.feature_generation import generate_features
from src_utils.geometry_utils import fix_coords_line, scale, fix_coords
from src_utils.graph_utils import find_cyclic_objects
from src_utils.heuristics_utils import get_closed_objects, get_bbox_circle_objects, get_bbox_rectangles_objects, \
    filter_grids, find_grid_lines_of_object, filter_by_line_number, find_closest_grid_to_center, find_next_grid_lines, \
    extend_grid_lines, limit_grid_line_by_color, limit_grid_line_by_dots, group_grids, filter_grids_duplicates, \
    filter_by_grids_areas, most_frequent, extend_short_gridlines, get_close_objects_based_on_groups, \
    add_color_closed_objects_groups, filter_duplicate_bboxes, filter_key_plan_grids
from src_utils.lines_processing import filter_by_color_nin, filter_lines_by_length, del_full_lines_inside_objects
from src_logging.log_config import setup_logger
from itertools import chain
import numpy as np
import pandas as pd
from fastapi.exceptions import HTTPException

logger = setup_logger(__name__)

def model_predict(data, model, threshold=None,
           thr_ratio=0.33, grids_probable_perc=0.5
                  ):
    X = data[model.feature_name()]
    probabilities = model.predict(X)
    if not threshold:
        num_possible_grids = int(X['num_objects'].values[0]*grids_probable_perc)
        for_threshold = np.sort(probabilities)[::-1][:num_possible_grids]
        while True:
            for_threshold = np.sort(for_threshold)[::-1]
            for_threshold_max, for_threshold_min = for_threshold.max(), for_threshold.min()
            if not for_threshold_min/for_threshold_max>thr_ratio:
                for_threshold = list(for_threshold)
                for_threshold.remove(for_threshold_min)
            else:
                break
        threshold = for_threshold.min()
        threshold = float(str(threshold)[:3])
    prediction = (model.predict(X)>=threshold).astype(int)
    return prediction

def find_grid_lines(lines,
                    cubic_lines,
                    circles,
                    rectangles,
                    quads,
                    parsed_text,
                    model,
                    config,
                    grids_areas:list=None
                   ):
    # get data
    try:
        lines_attributes = lines['lines_attributes']
        svg_width, svg_height = lines['svg_width'], lines['svg_height']
        lines = lines['lines_data']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with lines data : {str(e)}')

    try:
        circles_attributes = circles['attributes']
        circles = circles['circles_data']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with circles data : {str(e)}')

    try:
        rectangles_attributes = rectangles['attributes']
        rectangles = rectangles['rectangles_data']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with rectangles data : {str(e)}')

    try:
        quads_attributes = quads['attributes']
        quads_objects = quads['quad_bezier_lines_data']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with quads data : {str(e)}')

    try:
        cub_bezier_lines, cub_attributes = cubic_lines['cub_bezier_lines_data'], cubic_lines['attributes']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with cubic data : {str(e)}')

    try:
        parsed_text = parsed_text['parsed_text']
    except Exception as e:
        raise HTTPException(404, detail=f'Issue with parsed text data : {str(e)}')


    logger.info('Unravelled svg elements data and text')
    # get closed objects based on groups
    closed_objects_groups = get_close_objects_based_on_groups(lines, lines_attributes,
                                                              width=svg_width,
                                                              height=svg_height,
                                                              **config['get_close_objects_based_on_groups'])
    logger.info('Found closed objects based on groups')
    # filter by color
    lines += cub_bezier_lines
    lines_attributes += cub_attributes

    # get attributes dicts
    circles_attributes_dict = dict([((circles[c]['center']['x'], circles[c]['center']['y'], circles[c]['radius']), \
                                     i) \
                                    for c, i in enumerate(circles_attributes)])

    lines_attributes_dict = dict([(tuple(lines[c]), \
                                   i) \
                                  for c, i in enumerate(lines_attributes)])

    rectangles_attributes_dict = dict([(tuple(fix_coords(rectangles[c])), i) \
                                       for c, i in enumerate(rectangles_attributes)])

    quads_attributes_dict = dict([(tuple(quads_objects[c]['bbox'].values()), i) \
                                  for c, i in enumerate(quads_attributes)])

    #create attributes dict and
    data_attr = {}
    for i_, line_TEPP in enumerate(lines):
        key_TEMP1 = ','.join([str(elem) for elem in line_TEPP])
        key_TEMP2 = ','.join([str(elem) for elem in line_TEPP[2:] + line_TEPP[:2]])
        #if already exist color for line, write valid
        if key_TEMP1 in data_attr.keys():
            if data_attr[key_TEMP1] not in [None, '#FFFFFF']:
                continue
        if key_TEMP2 in data_attr.keys():
            if data_attr[key_TEMP2] not in [None, '#FFFFFF']:
                continue  
        data_attr[key_TEMP1] = lines_attributes[i_]['color']
        data_attr[key_TEMP2] = lines_attributes[i_]['color']

    for i_, ent in enumerate(circles):
        circle_lines = ent['lines']
        xs = list(chain(*[[i[0], i[2]] for i in circle_lines]))
        ys = list(chain(*[[i[1], i[3]] for i in circle_lines]))
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        key_TEMP = ','.join([str(elem) for elem in [min_x, min_y, max_x, max_y]])
        data_attr[key_TEMP] = circles_attributes[i_]['color']
        for line_KEK in circle_lines:
            key_TEMP = ','.join([str(elem) for elem in line_KEK])
            data_attr[key_TEMP] = circles_attributes[i_]['color']

    for i_, ent in enumerate(rectangles):
        key_TEMP = ','.join([str(elem) for elem in ent])
        data_attr[key_TEMP] = rectangles_attributes[i_]['color']
        key_TEMP = ','.join([str(elem) for elem in ent[2:] + ent[:2]])
        data_attr[key_TEMP] = rectangles_attributes[i_]['color']

    for i_, ent in enumerate(quads_objects):
        quads_objects_lines = ent['lines']
        
        xs = list(chain(*[[i[0], i[2]] for i in quads_objects_lines]))
        ys = list(chain(*[[i[1], i[3]] for i in quads_objects_lines]))
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        key_TEMP = ','.join([str(elem) for elem in [min_x, min_y, max_x, max_y]])
        data_attr[key_TEMP] = quads_attributes[i_]['color']
        for line_KEK in quads_objects_lines:
            key_TEMP = ','.join([str(elem) for elem in line_KEK])
            data_attr[key_TEMP] = quads_attributes[i_]['color']
     
    lines = filter_by_color_nin(lines, lines_attributes)
    circles = filter_by_color_nin(circles, circles_attributes)
    rectangles = filter_by_color_nin(rectangles, rectangles_attributes)
    quads_objects = filter_by_color_nin(quads_objects, quads_attributes)

    # copy original lines
    lines_original = lines.copy()


    logger.info('Applied filtration by color')
    # normalize liens
    lines = [fix_coords_line(i) for i in lines]
    logger.info('Normalized lines')

    # find cyclic objects
    cyclic_objects = find_cyclic_objects(lines)
    closed_objects = get_closed_objects(cyclic_objects, lines,
                                        svg_width, svg_height)
    
    logger.info('closed_objectsO')
    logger.info([list(i['bbox'].values()) for i in closed_objects])
    #logger.info([(lines[i], lines_attributes[i]) for i in range(lines.__len__()) if (lines[i][:2] == (2094, 550) or lines[i][2:] == (2094, 550))])
    #logger.info([(lines[i], lines_attributes[i]) for i in range(lines.__len__()) if (lines[i][:2] == [2094, 550] or lines[i][2:] == [2094, 550])])
    logger.info('closed_objects')
    # add color to closed_objects_groups
    closed_objects_groups = add_color_closed_objects_groups(closed_objects_groups,
                                    lines=lines,
                                    lines_attributes_dict=lines_attributes_dict)

    # add color to closed_objects
    for i in closed_objects:
        for j in i['lines']:
            attr = lines_attributes_dict.get(tuple(j))
            if not attr:
                attr = lines_attributes_dict.get(tuple([*j[2:], *j[:2]]))

            i['color'] = attr['color']
            break
    # get circle objects
    circles_objects = get_bbox_circle_objects(circles, circles_attributes_dict)

    # get rectangles objects
    rectangles_objects = get_bbox_rectangles_objects(rectangles, rectangles_attributes_dict)

    # get color for quads
    for i in quads_objects:
        i['color'] = quads_attributes_dict[tuple(i['bbox'].values())]['color']

    # get everything togather
    closed_objects += rectangles_objects + circles_objects + quads_objects
    logger.info('Got closed objects')

    # reduce number of lines checks
    lines = set(lines)
    object_lines = list(chain(*[i['lines'] for i in closed_objects]))
    object_lines = set(map(fix_coords_line, object_lines))
    lines = list(lines.difference(object_lines))
    closed_objects+=closed_objects_groups
    logger.info('LINES2')
    logger.info([(lines[i], lines_attributes[i]) for i in range(lines.__len__()) if (lines[i][:2] == (2094, 550) or lines[i][2:] == (2094, 550))])
    logger.info([(lines[i], lines_attributes[i]) for i in range(lines.__len__()) if (lines[i][:2] == [2094, 550] or lines[i][2:] == [2094, 550])])
    logger.info('LINES2')

    # get concat of parsed text
    parsed_text_concat = []
    for bbox_text_one in parsed_text:
        coords = list(map(lambda x: int(np.round(x)),
                          [bbox_text_one['x0'], bbox_text_one['y0'], bbox_text_one['x1'], bbox_text_one['y1']]))
        text_message = ' '.join([i['message'] for i in bbox_text_one['spans']])
        parsed_text_concat.append([text_message, coords, bbox_text_one])

    # filter girds
    list_of_boxes_cand, dict_text = filter_grids(closed_objects, parsed_text_concat,
                                                 width=svg_width,
                                                 height=svg_height,
                                                 **config['filter_grids'])
    logger.info(f'Filtered grids by text info and size, number of grids : {len(list_of_boxes_cand)}')

    # filter lines
    lines = filter_lines_by_length(lines, svg_width, svg_height)
    logger.info('LINES2.1')
    logger.info([(lines[i], lines_attributes[i]) for i in range(lines.__len__()) if (lines[i][:2] == (2094, 550) or lines[i][2:] == (2094, 550))])
    logger.info([(lines[i], lines_attributes[i]) for i in range(lines.__len__()) if (lines[i][:2] == [2094, 550] or lines[i][2:] == [2094, 550])])
    logger.info('LINES2.1')
    # del full lines inside objects
    objects_coords = [list(i['bbox'].values()) for i in closed_objects]
    logger.info('objects_coords')
    logger.info(objects_coords)
    logger.info('objects_coords')          
    to_del = del_full_lines_inside_objects(objects_coords, lines)
                       
    lines = list(set(lines).difference(to_del))
    logger.info(f'Number of lines : {len(lines)}')
    logger.info('LINES3')
    logger.info([(lines[i], lines_attributes[i]) for i in range(lines.__len__()) if (lines[i][:2] == (2094, 550) or lines[i][2:] == (2094, 550))])
    logger.info([(lines[i], lines_attributes[i]) for i in range(lines.__len__()) if (lines[i][:2] == [2094, 550] or lines[i][2:] == [2094, 550])])
    logger.info('LINES3')
    # find first line
    bbox_lines_dict, lines_dict = find_grid_lines_of_object(list_of_boxes_cand, lines,
                                                **config['find_grid_lines_of_object'])
    logger.info('Found start of new grid lines')


    # filter duplicates by bboxes
    bboxes_to_filter = filter_duplicate_bboxes(list(bbox_lines_dict.keys()),
                                               **config['filter_duplicate_bboxes'])
    for i in bboxes_to_filter:
        try:
            del bbox_lines_dict[i]
        except:
            pass


    # filter grids by number of connections
    if grids_areas:
        grids_width, grids_height = grids_areas[0]['width'], grids_areas[0]['height']
        grids_areas = [i['data'] for i in grids_areas]
        grids_areas = [[i['x_min'], i['y_min'], i['x_max'], i['y_max']] for i in grids_areas]
        if (grids_width != svg_width or grids_height != svg_height)\
                and not (grids_width==svg_height and grids_height==svg_width):
            grids_areas = [list(map(lambda x: int(np.round(x)),
                                    scale(i, original_size=(grids_width, grids_height),
                                          new_size=(svg_width, svg_height)))) for i in grids_areas]
        else:
            grids_areas = [list(map(lambda x: int(np.round(x)),
                                    i)) for i in grids_areas]
        filtered_grids = filter_by_grids_areas(bbox_lines_dict, grids_areas, 
                                               **config['filter_by_grids_areas'])
        filtered_grids = [{'bbox': i, 'text': dict_text[i]} for i in filtered_grids]

    else:
        filtered_grids = filter_by_line_number(bbox_lines_dict, **config['filter_by_line_number'])
        logger.info('Filtered grids')

        # associate text data
        filtered_grids = [{'bbox': i, 'text': dict_text[i]} for i in filtered_grids]
        closed_objects_attributes = {}
        for i in closed_objects:
            try:
                closed_objects_attributes[tuple(i['bbox'].values())] = i['color']
            except Exception as e:
                logger.info(str(e))

        # associate color
        for i in filtered_grids:
            i['color'] = closed_objects_attributes[i['bbox']]

        # generate features
        logger.info('S_C')
        logger.info([i['color'] for i in filtered_grids])
        logger.info('N_C')
        #logger.info([i['color_'] for i in filtered_grids])
        #logger.info('E_C')
        filtered_grids = [i for i in filtered_grids if i['color'] is not None]
        
        logger.info('S_C')
        logger.info([i['color'] for i in filtered_grids])
        logger.info('E_C')
        
        features_df = generate_features(filtered_grids, width=svg_width, height=svg_height)

        # predict based on features
        prediction = model_predict(features_df, model,
                    threshold=config['model_filtering']['threshold'],
                                   thr_ratio=config['model_filtering']['thr_ratio'],
                                   grids_probable_perc=config['model_filtering']['grids_probable_perc'])

        filtered_grids = [i for c, i in enumerate(filtered_grids) if prediction[c]]

        logger.info('Filtered grids by model')

    logger.info(f'Number of grids after model filtration : {len(filtered_grids)}')
    # key-plan filtration
    filtered_grids = filter_key_plan_grids(grids=filtered_grids,
                          parsed_text=parsed_text,
                          lines=lines_original,
                          svg_width=svg_width,
                          svg_height=svg_height,
                          **config['filter_key_plan_grids'])
    logger.info(f'Number of grids after key-plan filtration : {len(filtered_grids)}')
    # line extension
    for i in filtered_grids:
        i['grid_lines'] = bbox_lines_dict[i['bbox']]

    for i in filtered_grids:
        i['text'] = ''.join([j['message'] for j in i['text']['spans']])

    for i in filtered_grids:
        i['center'] = (int(i['bbox'][0] + (i['bbox'][2] - i['bbox'][0]) / 2), int(i['bbox'][1] + (i['bbox'][3] - i['bbox'][1]) / 2))


    to_del_grids = []
    for c, i in enumerate(filtered_grids):
        list_color_cand = [data_attr[','.join([str(elem) for elem in ii])]  for ii in lines_dict[i['bbox']]]
        list_color_cand = [z for z in list_color_cand if z]+[i['color']]
        if not list_color_cand:
            to_del_grids.append(c)
        else:
            i['color_'] = most_frequent(list_color_cand)

    filtered_grids = [i for c,i in enumerate(filtered_grids) if c not in to_del_grids]
    logger.info(f'Number of girds after filtration by colors : {len(filtered_grids)}')
    # Select grid that go closest to center of grid head
    filtered_grids = find_closest_grid_to_center(filtered_grids, data_attr,
                                **config['find_closest_grid_to_center'])
    # Check logic for continous not staight lines
    filtered_grids = find_next_grid_lines(filtered_grids,
                         data_attr,
                         lines)
    # Create extended lines
    filtered_grids = extend_grid_lines(filtered_grids)
    logger.info('Found extended grid lines')
    lines_original = filter_lines_by_length(lines_original, svg_width=svg_width,
                                            svg_height=svg_height)

    # Find lines which is part of our extended gridlines templates and have same color
    filtered_grids = limit_grid_line_by_color(lines_original, filtered_grids, data_attr,
                    **config['limit_grid_line_by_color'])
    filtered_grids = limit_grid_line_by_dots(filtered_grids,
                            **config['limit_grid_line_by_dots'])
    logger.info('Limited grid lines')
    filtered_grids = extend_short_gridlines(filtered_grids, svg_height, svg_width)
    logger.info('Extended short grid lines')
    filtered_grids = group_grids(filtered_grids)
    filtered_grids = filter_grids_duplicates(filtered_grids)
    logger.info('Filtered grid lines by duplicates')
    final_grids = []
    for i in filtered_grids:
        final_grids.append({'bbox': list(map(int,i['bbox'])),
                            'text': i['text'],
                            'grid': list(map(int,i['grid_line_F']))})
    return final_grids
