import math
from collections import Counter
from functools import partial

import shapely
from sklearn.metrics import pairwise_distances, euclidean_distances
from tqdm import tqdm
from src_utils.geometry_utils import get_bbox_lines, iou, get_n_closest, center_start_end_point_dist, check_line_type, \
    line_v_h_intersection, find_vect_direct_rads, create_vector_in_direction, line_v_h_o_intersection, \
    compute_slope_intercept, fix_coords, euclidean_dist, is_part_of_other, get_line_length, \
    rectangle_inside_rectangle, center_coord, fix_coords_line, bbox_line_intersection, is_point_inside_bbox
from src_utils.geometry_utils import is_point_inside_rect
from shapely.geometry import LineString
import re
import numpy as np
from itertools import chain, combinations
from operator import add
from src_utils.graph_utils import find_next_lines_for_grids, merge_small_lines_all
from copy import deepcopy
import pandas as pd
import logging
from src_logging import log_config
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.DEBUG)

def check_if_numeric(x):
    try:
        float(x)
        return True
    except:
        return False

def filter_by_line_number(bbox_lines_dict, max_conn=5):
    possible_grids = []
    for k,v in bbox_lines_dict.items():
        if len(set(map(tuple, v)))<max_conn:
            possible_grids.append(k)
    return possible_grids

def filter_by_grids_areas(bbox_lines_dict, grids_areas,
                          iou_thr=0.9):
    possible_grids = []
    for k,v in bbox_lines_dict.items():
        for area in grids_areas:
            if iou(area, k)>=iou_thr:
                possible_grids.append(k)
                break
    return possible_grids

def filter_by_size(bbox, width, height):
    x_dist = abs(bbox[0] - bbox[2])
    y_dist = abs(bbox[1] - bbox[3])
    area = int(x_dist * y_dist)
    overall_area = width * height
    area = area / overall_area
    return (area > 0.45), (area < 1e-5), (y_dist / height < 1e-3), (x_dist / width < 1e-3)

def filter_by_area_bigger(bbox, width, height):
    return [filter_by_size(bbox, width, height)[0]]

def filter_duplicate_bboxes(bboxes,
                            area_ratio_thr=0.95):
    to_remove = []
    for i in range(len(bboxes)):
        bbox_i = bboxes[i]
        area_i = abs(bbox_i[2] - bbox_i[0]) * abs(bbox_i[3] - bbox_i[1])
        for j in range(i + 1, len(bboxes)):
            bbox_j = bboxes[j]
            area_j = abs(bbox_j[2] - bbox_j[0]) * abs(bbox_j[3] - bbox_j[1])
            if rectangle_inside_rectangle(bbox_i, bbox_j, 5) and \
                    min([area_i, area_j]) / max([area_i, area_j]) > area_ratio_thr:
                to_rm = min([(bbox_i, area_i), (bbox_j, area_j)],
                            key=lambda x: x[1])[0]
                to_remove.append(to_rm)
    return to_remove


def add_color_closed_objects_groups(closed_objects_groups,
                                    lines,
                                    lines_attributes_dict):
    to_del = []
    for c, i in enumerate(closed_objects_groups):
        bbox = list(i['bbox'].values())
        colors = []
        lines_in_bbox = []
        for l in lines:
            if is_point_inside_bbox(l[:2], bbox) and is_point_inside_bbox(l[2:], bbox):
                attr = lines_attributes_dict.get(tuple(l), {})
                if not attr:
                    attr = lines_attributes_dict.get(tuple([*l[2:], *l[:2]]), {})

                color = attr.get('color')
                if color:
                    lines_in_bbox.append(l)
                    colors.append(color)

        if not colors:
            to_del.append(c)
        else:

            i['color'] = most_frequent(colors)
            i['lines'] = lines_in_bbox

    closed_objects_groups = [i for c, i in enumerate(closed_objects_groups) if c not in to_del]
    return closed_objects_groups


def get_close_objects_based_on_groups(lines, lines_attributes,
                                      width, height,
                                      tol=5):
    closed_objects = []
    groups = [i['group_id'] for i in lines_attributes]
    df = pd.DataFrame(list(zip(lines, groups)), columns=['lines', 'groups'])
    df = df.groupby('groups').agg({'lines': list})
    df = df[df['lines'].apply(len) > 2]
    for i in tqdm(df.iterrows()):
        tmp_lines = np.array(i[1]['lines'])
        bbox = [tmp_lines[:, [0, 2]].min(), tmp_lines[:, [1, 3]].min(), \
                tmp_lines[:, [0, 2]].max(), tmp_lines[:, [1, 3]].max()]
        if abs(bbox[2] - bbox[0]) >= tol and abs(bbox[3] - bbox[1]) >= tol and \
                not any(filter_by_size(bbox, width, height)):
            closed_objects.append({'bbox': {'x_min': bbox[0], 'y_min': bbox[1], 'x_max': bbox[2], \
                                            'y_max': bbox[3]},
                                   'lines': tmp_lines.tolist()})

    return closed_objects

def pairwise_delete_objects_by_area(objects1, objects2,
                                    iou_thr=0.2):
    to_del_cands = []
    for ent1 in objects1:
        detected_bbox1 = list(ent1['bbox'].values())
        area_detected1 = abs((detected_bbox1[3]-detected_bbox1[1])*(detected_bbox1[2]-detected_bbox1[0]))
        for c, ent2 in enumerate(objects2):
            detected_bbox2 = list(ent2['bbox'].values())
            area_detected2 = abs((detected_bbox2[3]-detected_bbox2[1])*(detected_bbox2[2]-detected_bbox2[0]))
            if iou(detected_bbox2, detected_bbox1)>iou_thr and area_detected1>area_detected2:
                to_del_cands.append(c)
    return to_del_cands
def get_closed_objects(possible_cycles, lines,
                       svg_width, svg_height,
                      tol=5,
                       filter_func=filter_by_size):   # Generate bbox for cyclic objects
    bboxes = []
    lines_bboxes = []
    for i in tqdm(possible_cycles):
        to_comp_x = []
        to_comp_y = []
        for j in i:
            start, end = j
            to_comp_x.extend([start[0], end[0]])
            to_comp_y.extend([start[1], end[1]])

        min_x, max_x = min(to_comp_x), max(to_comp_x)
        min_y, max_y = min(to_comp_y), max(to_comp_y)
        if abs(min_x - max_x) > tol and abs(min_y - max_y) > tol:
            bbox = [min_x, min_y, max_x, max_y]
            flags = filter_func(bbox, svg_width, svg_height)
            if not any(flags):
                bbox_lines = get_bbox_lines(bbox)
                points_intersections = set()
                line_intersect = False
                for bbox_line in bbox_lines:
                    for j in i:
                        start, end = j
                        do_intersect = LineString([start, end]) \
                            .intersection(LineString([bbox_line[:2], bbox_line[2:]]))

                        if do_intersect:
                            if isinstance(do_intersect, LineString):
                                line_intersect = True
                            else:
                                points_intersections.update(list(do_intersect.coords))

                if len(points_intersections) > 3 or line_intersect:
                    if bbox not in bboxes:
                        bboxes.append(bbox)
                        lines_bboxes.append(i)
            if flags[0]:
                for j in i:

                    if [*j[0], *j[1]] in lines:
                        lines.remove([*j[0], *j[1]])
                    elif [*j[1], *j[0]] in lines:
                        lines.remove([*j[1], *j[0]])


    lines_bboxes = [[[*j[0], *j[1]] for j in i] for i in lines_bboxes]
    bboxes_dict = []

    for c, i in enumerate(bboxes):
        bboxes_dict.append({'bbox': {'x_min': i[0], 'y_min': i[1], 'x_max': i[2], \
                                     'y_max': i[3]},
                            'lines': lines_bboxes[c],
                            'width': int(svg_width),
                            'height': int(svg_height)})


    return bboxes_dict


def merge_bboxes(bboxes_dict):
    new_objects = deepcopy(bboxes_dict)
    for i in range(len(bboxes_dict)):
        for j in range(i + 1, len(bboxes_dict)):

            bbox_i = list(bboxes_dict[i]['bbox'].values())
            bbox_i_lines = bboxes_dict[i]['lines']
            bbox_j = list(bboxes_dict[j]['bbox'].values())
            bbox_j_lines = bboxes_dict[j]['lines']

            flag1 = len(set(list(map(tuple, bbox_i_lines))) \
                        .intersection(list(map(tuple, bbox_j_lines))))
            if flag1:
                try:
                    new_objects.remove(bboxes_dict[i])
                    new_objects.remove(bboxes_dict[j])
                except:
                    pass

                new_objects.append({'bbox': dict(zip(list(bboxes_dict[i]['bbox'].keys()), [min(bbox_i[0], bbox_j[0]),
                                                                                           min(bbox_i[1], bbox_j[1]),
                                                                                           max(bbox_i[2], bbox_j[2]),
                                                                                           max(bbox_i[3], bbox_j[3])])),
                     'lines': list(set(list(map(tuple, bboxes_dict[i]['lines'] + bboxes_dict[j]['lines'])))),
                                    })

    return new_objects

def filter_closed_objects(closed_objects, tol=5):
    filtered = []
    for i in closed_objects:
        if abs(i['bbox']['x_max']-i['bbox']['x_min'])>tol and \
        abs(i['bbox']['y_max']-i['bbox']['y_min'])>tol:
            filtered.append(i)
    return filtered


def merge_groups(groups):
    merged_groups = {}

    def get_values(merged_groups):
        return list(chain(*[[k, *v] for k, v in merged_groups.items()]))

    def _merge_groups(group_key, merged_group_key):
        if group_key in groups:
            values = groups[group_key]
            if merged_group_key not in merged_groups:
                merged_groups[merged_group_key] = []
            for value in values:
                _merge_groups(value, merged_group_key)
                merged_groups[merged_group_key].append(value)

    for key in groups:
        if key not in get_values(merged_groups):
            _merge_groups(key, key)
    return merged_groups


def merge_bboxes_sld(bboxes_dict):
    new_objects = deepcopy(bboxes_dict)
    groups = {}
    for i in range(len(bboxes_dict)):
        group = []
        for j in range(i + 1, len(bboxes_dict)):

            bbox_i = list(bboxes_dict[i]['bbox'].values())
            bbox_i_lines = get_bbox_lines(bbox_i)
            bbox_j = list(bboxes_dict[j]['bbox'].values())
            bbox_j_lines = get_bbox_lines(bbox_j)

            flag1 = len(set(list(map(tuple, bbox_i_lines))) \
                        .intersection(list(map(tuple, bbox_j_lines))))

            if flag1 and bbox_i[1] == bbox_j[1] and bbox_i[3] == bbox_j[3]:
                group.append(j)
        if group:
            groups[i] = group

    groups = merge_groups(groups)
    for i, group in groups.items():
        if len(group) == 1:
            group.append(i)
            for idx in group:
                try:
                    new_objects.remove(bboxes_dict[idx])
                except:
                    pass

            bboxes_dict_i, bboxes_dict_j = bboxes_dict[group[0]], bboxes_dict[group[1]]
            bbox_i = list(bboxes_dict_i['bbox'].values())
            bbox_j = list(bboxes_dict_j['bbox'].values())
            new_objects.append( \
                {'bbox': dict(zip(list(bboxes_dict_i['bbox'].keys()), [min(bbox_i[0], bbox_j[0]),
                                                                       min(bbox_i[1], bbox_j[1]),
                                                                       max(bbox_i[2], bbox_j[2]),
                                                                       max(bbox_i[3], bbox_j[3])])),
                 'lines': list(set(list(map(tuple, bboxes_dict_i['lines'] + bboxes_dict_j['lines']))))
                 })

    return new_objects
def filter_grids(bboxes_dict, list_ok_text_reg, width, height,
                 shape_coef_min=1 / 1.5, shape_coef_max=1.5,
                 iou_thr=0.5,
                 regexes=None):
    def _filter_by_regex(text, patterns):
        return any([bool(re.match(pat, text)) for pat in patterns])

    if not regexes:
        regexes = []

    # filters grids by regex and size

    list_of_boxes_cand_ = []
    list_sucs_texts_ = []
    for ent in bboxes_dict:
        bbox = ent['bbox']
        # filtering by size
        area_of_box_in_proc = 100 * (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min']) / (width * height)

        if area_of_box_in_proc == .0:
            continue

        # filtering by shape
        shape_coef = (bbox['x_max'] - bbox['x_min']) / (bbox['y_max'] - bbox['y_min'])
        if shape_coef_min < shape_coef < shape_coef_max:
            text_inside = []
            for text_message_, text_coord_, text_ent in list_ok_text_reg:
                bb_coord_def = list(bbox.values())
                if iou(bb_coord_def, text_coord_) > iou_thr:
                    if len(text_inside) > 1:
                        break
                    text_message_T = text_message_[:]
                    text_coord_T = text_coord_[:]
                    text_inside.append([text_message_T, text_coord_T, text_ent])

            if len(text_inside) == 1:  # logic that only one proper text can be inside
                list_of_boxes_cand_.append(ent)
                list_sucs_texts_.append(text_inside[0])

    list_of_boxes_cand = []
    dict_text = {}
    for tesxts_, bboxes_ in zip(list_sucs_texts_, list_of_boxes_cand_):
        text_message_T, text_coord_T, text_ent = tesxts_
        # apply regex
        if check_if_numeric(text_message_T):
            result = True
        else:
            result = _filter_by_regex(text_message_T, regexes)

        if result:
            list_of_boxes_cand.append(bboxes_)
            key_bbx = tuple(bboxes_['bbox'].values())  #: 542, 'y_min': 1747, 'x_max': 599, 'y_max': 1813}
            dict_text[key_bbx] = text_ent
    return list_of_boxes_cand, dict_text


def find_grid_lines_of_object(list_of_boxes_cand, lines,
                              tol=4, top_n_to_cluster=50):
    # find grid lines of object
    bbox_lines_dict = {}
    lines_dict = {}

    for bbox_ in tqdm(list_of_boxes_cand):
        obj = tuple(bbox_['bbox'].values())
        lines_to_tranfer = bbox_['lines']
        bbox_sides = get_bbox_lines(obj)
        to_check_lines = get_n_closest(bbox_sides, lines,
                                       center_start_end_point_dist, n=top_n_to_cluster)
        tmp_lines = []
        for idx, side_bbox in enumerate(bbox_sides):
            side_bbox_type = check_line_type(side_bbox)
            for line in to_check_lines[idx]:
                start, end = line[:2], line[2:]
                line_type = check_line_type(line)
                if line_type != side_bbox_type:
                    if line_type != 'other':
                        if side_bbox_type == 'vertical':
                            flag, point = line_v_h_intersection(side_bbox, line)
                            if flag:
                                to_save = np.array([end, start])[
                                    np.argsort([np.sum(np.abs(np.array(point) - np.array(j))) \
                                                for j in [end, start]])]
                                tmp_lines.append([*to_save[0], *to_save[1]])
                            elif side_bbox[1] <= end[1] <= side_bbox[3] and abs(side_bbox[0] - end[0]) <= tol:
                                tmp_lines.append([*end, *start])
                            elif side_bbox[1] <= start[1] <= side_bbox[3] and abs(side_bbox[0] - start[0]) <= tol:
                                tmp_lines.append(line)


                        else:
                            flag, point = line_v_h_intersection(side_bbox, line)
                            if flag:
                                to_save = np.array([end, start])[
                                    np.argsort([np.sum(np.abs(np.array(point) - np.array(j))) \
                                                for j in [end, start]])]
                                tmp_lines.append([*to_save[0], *to_save[1]])

                            elif side_bbox[0] <= end[0] <= side_bbox[2] and abs(side_bbox[1] - end[1]) <= tol:
                                tmp_lines.append([*end, *start])
                            elif side_bbox[0] <= start[0] <= side_bbox[2] and abs(side_bbox[1] - start[1]) <= tol:
                                tmp_lines.append(line)



                    elif line_type == 'other':
                        # check intersection between lines and vector from the point
                        try:
                            rad = find_vect_direct_rads(line)
                        except:
                            pass

                        if tol:

                            end_point1 = create_vector_in_direction(start, rad, -tol)
                            end_point2 = create_vector_in_direction(end, rad, tol)

                            vector_start = [*start, *end_point1]
                            vector_end = [*start, *end_point2]
                            m, b = compute_slope_intercept(vector_end)
                            flag, point = line_v_h_o_intersection(side_bbox, [*vector_end, m, b])
                            if flag:
                                tmp_lines.append(line)
                            else:
                                m, b = compute_slope_intercept(vector_start)
                                flag, point = line_v_h_o_intersection(side_bbox, [*vector_start, m, b])

                                if flag:
                                    tmp_lines.append(line)
                        else:
                            m, b = compute_slope_intercept(line)
                            flag, point = line_v_h_o_intersection(side_bbox, [*line, m, b])
                            if flag:
                                tmp_lines.append(line)
        if tmp_lines:
            bbox_lines_dict[obj] = tmp_lines
            lines_dict[obj] = deepcopy(lines_to_tranfer)
    return bbox_lines_dict, lines_dict

def get_bbox_circle_objects(circles, circles_attributes_dict=None):
    circles_objects = []
    for ent in circles:
        circle_lines = ent['lines']
        xs = list(chain(*[[i[0], i[2]] for i in circle_lines]))
        ys = list(chain(*[[i[1], i[3]] for i in circle_lines]))
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        d = {'bbox': dict(zip(['x_min', 'y_min', 'x_max', 'y_max'], [min_x, min_y, max_x, max_y])),
                                'lines': circle_lines}
        if circles_attributes_dict:
            d['color'] = circles_attributes_dict[(ent['center']['x'], \
                                                                  ent['center']['y'],
                                                                  ent['radius'])]['color']
        circles_objects.append(d)

    return circles_objects

def filter_cyclic_objects(subgraphs, thr=2):
    filtered_subgraphs = []
    for i in tqdm(subgraphs):
        if len(i) > 3:
            center_coords = [center_coord(list(chain(*i))) for i in i]
            distances = euclidean_distances(center_coords)
            val = np.mean(np.sort(distances)[:, 1])
            if val > thr:
                filtered_subgraphs.append(i)
    return filtered_subgraphs

def find_closest_grid_to_center(filtered_grids_with_text, data_attr,
                 extend_factor=400):
    for nmu_obj, gdir in enumerate(filtered_grids_with_text):

        color_grid = gdir['color_']
        bb_center = gdir['center']
        rad_ = abs(gdir['bbox'][2]-gdir['bbox'][0])/2 # we will limit distance, so no more grab of random lines
        closest_cand_grid_dist = rad_/2#1000 #
        closest_cand_grid_dist_c = rad_/2
        closest_cand_grid_dist_emergency = 1000 #for casses when shit hapens
        closest_cand_grid = None
        if color_grid != -1:
            for L_ in gdir['grid_lines']:
                if data_attr[','.join([str(elem) for elem in L_])] == color_grid:
                    # L_ =
                    var_a_p = extend_line(L_, extend_factor)
                    var_a_n = extend_line(L_, -extend_factor)
                    var_b_p = extend_line_reverse(L_, extend_factor)
                    var_b_n = extend_line_reverse(L_, -extend_factor)
                    if euclidean_dist(bb_center, L_[2:]) > euclidean_dist(bb_center, L_[:2]):
                        if min(euclidean_dist(bb_center, var_a_p[:2]), euclidean_dist(bb_center, var_a_p[2:])) < min(
                                euclidean_dist(bb_center, var_a_n[:2]), euclidean_dist(bb_center, var_a_n[2:])):
                            ext_line = extend_line(L_, extend_factor)
                        else:
                            ext_line = extend_line(L_, -extend_factor)
                    else:
                        if min(euclidean_dist(bb_center, var_b_p[:2]), euclidean_dist(bb_center, var_b_p[2:])) < min(
                                euclidean_dist(bb_center, var_b_n[:2]), euclidean_dist(bb_center, var_b_n[2:])):
                            ext_line = extend_line_reverse(L_, extend_factor)
                        else:
                            ext_line = extend_line_reverse(L_, -extend_factor)
                    clst_p = point_closest(ext_line, bb_center)
                    if euclidean_dist(clst_p, bb_center) < closest_cand_grid_dist:
                        closest_cand_grid_dist = euclidean_dist(clst_p, bb_center)
                        closest_cand_grid = L_
        if closest_cand_grid == None:
            for L_ in gdir['grid_lines']:
                var_a_p = extend_line(L_, extend_factor)
                var_a_n = extend_line(L_, -extend_factor)
                var_b_p = extend_line_reverse(L_, extend_factor)
                var_b_n = extend_line_reverse(L_, -extend_factor)
                if euclidean_dist(bb_center, L_[2:]) > euclidean_dist(bb_center, L_[:2]):
                    if min(euclidean_dist(bb_center, var_a_p[:2]), euclidean_dist(bb_center, var_a_p[2:])) < min(
                            euclidean_dist(bb_center, var_a_n[:2]), euclidean_dist(bb_center, var_a_n[2:])):
                        ext_line = extend_line(L_, extend_factor)
                    else:
                        ext_line = extend_line(L_, -extend_factor)
                else:
                    if min(euclidean_dist(bb_center, var_b_p[:2]), euclidean_dist(bb_center, var_b_p[2:])) < min(
                            euclidean_dist(bb_center, var_b_n[:2]), euclidean_dist(bb_center, var_b_n[2:])):
                        ext_line = extend_line_reverse(L_, extend_factor)
                    else:
                        ext_line = extend_line_reverse(L_, -extend_factor)
                clst_p = point_closest(ext_line, bb_center)
                if euclidean_dist(clst_p, bb_center) < closest_cand_grid_dist_c:
                    closest_cand_grid_dist_c = euclidean_dist(clst_p, bb_center)
                    closest_cand_grid = L_

        if closest_cand_grid == None:
            for L_ in gdir['grid_lines']:
                var_a_p = extend_line(L_, extend_factor)
                var_a_n = extend_line(L_, -extend_factor)
                var_b_p = extend_line_reverse(L_, extend_factor)
                var_b_n = extend_line_reverse(L_, -extend_factor)
                if euclidean_dist(bb_center, L_[2:]) > euclidean_dist(bb_center, L_[:2]):
                    if min(euclidean_dist(bb_center, var_a_p[:2]), euclidean_dist(bb_center, var_a_p[2:])) < min(
                            euclidean_dist(bb_center, var_a_n[:2]), euclidean_dist(bb_center, var_a_n[2:])):
                        ext_line = extend_line(L_, extend_factor)
                    else:
                        ext_line = extend_line(L_, -extend_factor)
                else:
                    if min(euclidean_dist(bb_center, var_b_p[:2]), euclidean_dist(bb_center, var_b_p[2:])) < min(
                            euclidean_dist(bb_center, var_b_n[:2]), euclidean_dist(bb_center, var_b_n[2:])):
                        ext_line = extend_line_reverse(L_, extend_factor)
                    else:
                        ext_line = extend_line_reverse(L_, -extend_factor)
                clst_p = point_closest(ext_line, bb_center)
                if euclidean_dist(clst_p, bb_center) < closest_cand_grid_dist_emergency:
                    #print('s')
                    closest_cand_grid_dist_emergency = euclidean_dist(clst_p, bb_center)
                    closest_cand_grid = L_

        
        gdir['grid_lines_all_cand'] = gdir['grid_lines']
        gdir['grid_lines'] = closest_cand_grid
    return filtered_grids_with_text

def find_next_grid_lines(filtered_grids_with_text,
                         data_attr,
                         lines):
    for obj_ in filtered_grids_with_text:

        #if obj_['text'] == 'B.2': logger.info(f"INFO S")
        #logger.info(obj_['text'])
        #logger.info(obj_['grid_lines'])
        #logger.info(f"INFO E")
        
        start_line = obj_['grid_lines']

        lines_TEMP = [i for i in lines if data_attr[','.join([str(elem) for elem in i])] == data_attr[
            ','.join([str(elem) for elem in start_line])]]
        if obj_['text'] == 'B.2': 
            logger.info(f"INFO S TT_")
            TT_ = [i for i in lines if (i[:2] == (2094, 550) or i[2:] == (2094, 550))]
            TT_2 = [i for i in lines if (i[:2] == [2094, 550] or i[2:] == [2094, 550])]
            logger.info(TT_)
            logger.info(TT_2)
            logger.info(f"INFO TT_")
        next_lines = find_next_lines_for_grids(lines_TEMP, start_line, obj_['center'])
        if next_lines:
            try:
                #Check if first line longer than radius (R) and shorter then 5R
                R = (obj_['bbox'][2] - obj_['bbox'][0])/2
                #if obj_['text'] == 'B.2': logger.info('next_lines')
                #if obj_['text'] == 'B.2': logger.info(next_lines)
                try:
                    line_nstr = list(next_lines[0]) + list(next_lines[1])
                except Exception:
                    end_L_T = next_lines[0]
                    Wh_L_T = [i for i in lines_TEMP if (i[:2] == next_lines[0] or i[2:] == next_lines[0])][0]
                    st_L_T = Wh_L_T[:2] if Wh_L_T[2:] == end_L_T else Wh_L_T[2:]
                    line_nstr = end_L_T + st_L_T
                    #if obj_['text'] == 'B.2': logger.info('Wh_L_T')
                    #if obj_['text'] == 'B.2': logger.info(Wh_L_T)
                if get_line_length(line_nstr) < R or get_line_length(line_nstr) > 5*R:
                    print('cowabunga it is')
                    print(1/0)
                obj_['grid_lines'] = line_nstr
                obj_['line_is_CONT_STR'] = False
                #if obj_['text'] == 'B.2': logger.info(f"E1")
                
            except Exception:
                obj_['grid_lines'] = start_line
                obj_['line_is_CONT_STR'] = True
                #if obj_['text'] == 'B.2': logger.info(f"E2")
        else:
            obj_['grid_lines'] = start_line
            obj_['line_is_CONT_STR'] = True
            #if obj_['text'] == 'B.2': logger.info(f"E3")
        
        #logger.info(f"INFO SE")
        #logger.info(obj_['text'])
        #logger.info(obj_['grid_lines'])
        #if obj_['text'] == 'B.2': logger.info(f"INFO EE")
    return filtered_grids_with_text

def extend_grid_lines(filtered_grids_with_text):
    '''Extend start lines for grid line and selected with the longest distance from center of head gridline, to end'''
    for nmu_obj, gdir in enumerate(filtered_grids_with_text):
        L_ = gdir['grid_lines']
        bb_center = gdir['center']
        var_a_p = extend_line(L_, 400)
        var_a_n = extend_line(L_, -400)
        var_b_p = extend_line_reverse(L_, 400)
        var_b_n = extend_line_reverse(L_, -400)

        # DUMB BUT WORKS, REWRITE BETTER
        max_d_var_a_p = max(euclidean_dist(bb_center, var_a_p[:2]), euclidean_dist(bb_center, var_a_p[2:]))
        max_d_var_a_n = max(euclidean_dist(bb_center, var_a_n[:2]), euclidean_dist(bb_center, var_a_n[2:]))
        max_d_var_b_p = max(euclidean_dist(bb_center, var_b_p[:2]), euclidean_dist(bb_center, var_b_p[2:]))
        max_d_var_b_n = max(euclidean_dist(bb_center, var_b_n[:2]), euclidean_dist(bb_center, var_b_n[2:]))

        if max(max_d_var_a_p, max_d_var_a_n, max_d_var_b_p, max_d_var_b_n) == max_d_var_a_p:
            ext_line = var_a_p.copy()
        elif max(max_d_var_a_p, max_d_var_a_n, max_d_var_b_p, max_d_var_b_n) == max_d_var_a_n:
            ext_line = var_a_n.copy()
        elif max(max_d_var_a_p, max_d_var_a_n, max_d_var_b_p, max_d_var_b_n) == max_d_var_b_p:
            ext_line = var_b_p.copy()
        elif max(max_d_var_a_p, max_d_var_a_n, max_d_var_b_p, max_d_var_b_n) == max_d_var_b_n:
            ext_line = var_b_n.copy()

        gdir['ext_line'] = ext_line

    return filtered_grids_with_text


def find_uncovered_regions(small_lines, return_bounds=False):
    # find uncovered regions for sequence of lines, and return thus regions as new lines (aka inversion logic)
    result = []
    small_lines.sort(key=lambda x: x[0])  # Sort small_lines based on start points
    last_covered_end = small_lines[0][0]
    global_length = small_lines[-1][1]

    for start, end in small_lines:
        if start > last_covered_end:
            result.append([last_covered_end, start])
        last_covered_end = max(last_covered_end, end)

    if last_covered_end < global_length:
        result.append([last_covered_end, global_length])
    if not result:
        if return_bounds:
            return [[small_lines[0][0], small_lines[-1][1]]], small_lines[0], small_lines[-1]
        else:
            return [[small_lines[0][0], small_lines[-1][1]]]
    if return_bounds:
        return result, small_lines[0], small_lines[-1]
    return result


def filtrate_by_is_part_of_other(lines_in_greedline, orient):  # TODO add data_atr entr
    # To fix problem of line intersection (which doesn`t allow us correctly calculate distance outliers) we will remove
    # all lines, and create new, via double "inversion" of original lines
    if lines_in_greedline:
        lines_in_greedline_TEST = lines_in_greedline.copy()
        lines_in_greedline_TEST = [fix_coords_line(i) for i in lines_in_greedline_TEST]

        if orient == 'vertical':
            line_base = lines_in_greedline_TEST[0][0]  # TODO test for troubles
            small_lines = [[i[1], i[3]] for i in lines_in_greedline_TEST]
            uncovered_regions, first_line, last_line = find_uncovered_regions(small_lines, True)
            last_line = [uncovered_regions[-1][1], last_line[1]]
            first_line = [first_line[0], uncovered_regions[0][0]]
            uncovered_regions_2 = find_uncovered_regions(uncovered_regions)
            uncovered_regions_2.append(first_line)
            uncovered_regions_2.append(last_line)
            final_line = [[line_base, i[0], line_base, i[1]] for i in uncovered_regions_2]

        if orient == 'horizontal':
            line_base = lines_in_greedline_TEST[0][1]  # TODO test for troubles
            small_lines = [[i[0], i[2]] for i in lines_in_greedline_TEST]
            uncovered_regions, first_line, last_line = find_uncovered_regions(small_lines, True)
            last_line = [uncovered_regions[-1][1], last_line[1]]
            first_line = [first_line[0], uncovered_regions[0][0]]
            uncovered_regions_2 = find_uncovered_regions(uncovered_regions)
            uncovered_regions_2.append(first_line)
            uncovered_regions_2.append(last_line)
            final_line = [[i[0], line_base, i[1], line_base] for i in uncovered_regions_2]
        return final_line
    else:
        return -1 #if no lines in extended gridline we will use base gridline as only one element left

def limit_grid_line_by_color(lines_original, filtered_grids_with_text, data_attr,
                    tol=1):
    '''Finding candidates to be part of gridline, with same color, and being part of extended start line of gridline'''
    list_lines_vertical = [li_ for li_ in lines_original if check_line_type(li_) == 'vertical']
    list_lines_horizontal = [li_ for li_ in lines_original if check_line_type(li_) == 'horizontal']
    list_lines_other = [li_ for li_ in lines_original if check_line_type(li_) == 'other']
    line_eq_tol = 15
    angle_tol = .25
    for obj_ in filtered_grids_with_text:
        L_color = -1
        L_ = obj_['grid_lines']
        L_extended = obj_['ext_line']
        line_typ = check_line_type(L_extended)
        temp_L_ = ','.join([str(elem) for elem in L_])
        if temp_L_ in data_attr.keys():
            L_color = data_attr[temp_L_]

        if line_typ == 'vertical':
            list_lines_vertical_T = [i for i in list_lines_vertical if
                                     data_attr[','.join([str(elem) for elem in i])] == L_color]
            lines_in_greedline = [i for i in list_lines_vertical_T if is_part_of_other(i, L_extended, ['vertical'])]

            list_if_add_lines = []
            for ii in range(1, tol + 1):
                list_if_add_lines += [i for i in list_lines_vertical_T if
                                      is_part_of_other(i, list(map(add, L_extended, [ii, 0, ii, 0])), ['vertical'])]
                list_if_add_lines += [i for i in list_lines_vertical_T if
                                      is_part_of_other(i, list(map(add, L_extended, [ii * -1, 0, ii * -1, 0])),
                                                       ['vertical'])]
            lines_in_greedline += list_if_add_lines  #

            lines_in_greedline = filtrate_by_is_part_of_other(lines_in_greedline, 'vertical')
        elif line_typ == 'horizontal':
            list_lines_horizontal_T = [i for i in list_lines_horizontal if
                                       data_attr[','.join([str(elem) for elem in i])] == L_color]
            lines_in_greedline = [i for i in list_lines_horizontal_T if is_part_of_other(i, L_extended, ['horizontal'])]

            list_if_add_lines = []
            for ii in range(1, tol + 1):
                list_if_add_lines += [i for i in list_lines_horizontal_T if
                                      is_part_of_other(i, list(map(add, L_extended, [0, ii, 0, ii])), ['horizontal'])]
                list_if_add_lines += [i for i in list_lines_horizontal_T if
                                      is_part_of_other(i, list(map(add, L_extended, [0, ii * -1, 0, ii * -1])),
                                                       ['horizontal'])]
            lines_in_greedline += list_if_add_lines

            lines_in_greedline = filtrate_by_is_part_of_other(lines_in_greedline, 'horizontal')
        else:

            rads_b = [find_vect_direct_rads(L_extended), find_vect_direct_rads([*L_extended[2:], *L_extended[:2]])]
            L_extended_F = fix_coords_line(L_extended)
            rads_b_F = [find_vect_direct_rads(L_extended_F),  find_vect_direct_rads([*L_extended_F[2:], *L_extended_F[:2]])]
            lines_in_greedline = []
            for i in tqdm(range(len(list_lines_other))):
                if (data_attr[','.join([str(elem) for elem in list_lines_other[i]])] != L_color or
                        data_attr[','.join([str(elem) for elem in fix_coords_line(list_lines_other[i])])] != L_color):
                    continue
                to_app = False

                # DEF_LINE_EXT with DEF_LINE_CAND
                line_i = list_lines_other[i]
                rads_i = [find_vect_direct_rads(line_i), \
                          find_vect_direct_rads([*line_i[2:], *line_i[:2]])]
                if is_part_of_other(L_extended, line_i, [], rads1=rads_b, rads2=rads_i,
                                       line_eq_tol=line_eq_tol) or is_part_of_other(line_i, L_extended, [],
                                                                                       rads1=rads_i, rads2=rads_b,
                                                                                       line_eq_tol=line_eq_tol,
                                                                                       angle_tol=angle_tol):
                    to_app = True
                # DEF_LINE_EXT with FIXED_DEF_LINE_CAND

                line_i = list_lines_other[i]
                line_i = fix_coords_line(line_i)

                rads_i = [find_vect_direct_rads(line_i), \
                          find_vect_direct_rads([*line_i[2:], *line_i[:2]])]

                if is_part_of_other(L_extended, line_i, [], rads1=rads_b, rads2=rads_i,
                                       line_eq_tol=line_eq_tol) or is_part_of_other(line_i, L_extended, [],
                                                                                       rads1=rads_i, rads2=rads_b,
                                                                                       line_eq_tol=line_eq_tol,
                                                                                       angle_tol=angle_tol):
                    to_app = True


                # FIXED_DEF_LINE_EXT with DEF_LINE_CAND
                line_i = list_lines_other[i]
                rads_i = [find_vect_direct_rads(line_i), \
                          find_vect_direct_rads([*line_i[2:], *line_i[:2]])]
                if is_part_of_other(L_extended_F, line_i, [], rads1=rads_b_F, rads2=rads_i,
                                       line_eq_tol=line_eq_tol) or is_part_of_other(line_i, L_extended_F, [],
                                                                                       rads1=rads_i, rads2=rads_b_F,
                                                                                       line_eq_tol=line_eq_tol,
                                                                                       angle_tol=angle_tol):
                    to_app = True


                # FIXED_DEF_LINE_EXT with FIXED_DEF_LINE_CAND
                line_i = list_lines_other[i]
                line_i = fix_coords_line(line_i)
                rads_i = [find_vect_direct_rads(line_i), \
                             find_vect_direct_rads([*line_i[2:], *line_i[:2]])]
                if is_part_of_other(L_extended_F, line_i, [], rads1=rads_b_F, rads2=rads_i, line_eq_tol=line_eq_tol) or \
                        is_part_of_other(line_i, L_extended_F, [], rads1=rads_i, rads2=rads_b_F, line_eq_tol=line_eq_tol, angle_tol=angle_tol):
                    to_app = True

                if to_app:
                    lines_in_greedline.append(list_lines_other[i])


        if lines_in_greedline == -1:
            obj_['lines_in_greedline'] = [obj_['grid_lines']]
        else:
            obj_['lines_in_greedline'] = lines_in_greedline.copy()


    return filtered_grids_with_text

def limit_grid_line_by_dots(filtered_grids_with_text,
                            n_std=4, num_diag_max_allowe_dist = 1):
    for obj_ in filtered_grids_with_text:
        L_ = list(obj_['grid_lines'])
        rad_L = euclidean_dist(obj_['bbox'][:2], obj_['bbox'][2:]) / 2
        start_point = L_[:2] if euclidean_dist(obj_['center'], L_[:2]) < euclidean_dist(obj_['center'], L_[2:]) else L_[
                                                                                                                     2:]
        lines_inside = obj_['lines_in_greedline']
        if len(lines_inside) == 1:
            if max(get_line_length(start_point + list(lines_inside[0][:2])),
                   get_line_length(start_point + list(lines_inside[0][2:]))) == get_line_length(
                    start_point + list(lines_inside[0][:2])):
                grid_line = start_point + list(lines_inside[0][:2])
            else:
                grid_line = start_point + list(lines_inside[0][2:])
            obj_['grid_line_F'] = grid_line
            continue
        if len(lines_inside) == 0:
            grid_line = L_
            obj_['grid_line_F'] = grid_line
            continue
        grid_center_point = obj_['center']
        list_for_order = [max(euclidean_dist(i[:2], grid_center_point), euclidean_dist(i[2:], grid_center_point)) for i
                          in lines_inside]
        lines_inside_sorted_by_euc = [lines_inside for _, lines_inside in sorted(zip(list_for_order, lines_inside))]
        diff_euc = [min(euclidean_dist(i[:2], j[:2]), euclidean_dist(i[2:], j[:2]), euclidean_dist(i[:2], j[2:]),
                        euclidean_dist(i[2:], j[2:])) for i, j in
                    zip(lines_inside_sorted_by_euc[:-1], lines_inside_sorted_by_euc[1:])]

        std_diff_euc = min(np.median(diff_euc),
                           num_diag_max_allowe_dist * rad_L)
        if std_diff_euc == 0:
            grid_line = start_point + list(lines_inside_sorted_by_euc[-1][:2])
            obj_['grid_line_F'] = grid_line
            continue
        last_elem_id = np.where(diff_euc > (std_diff_euc * n_std))[0]

        # end metric_exp_space
        if len(last_elem_id):
            last_elem_id = last_elem_id[0]
        else:
            last_elem_id = -1
        if max(get_line_length(start_point + list(lines_inside_sorted_by_euc[last_elem_id][:2])),
               get_line_length(start_point + list(lines_inside_sorted_by_euc[last_elem_id][2:]))) == get_line_length(
                start_point + list(lines_inside_sorted_by_euc[last_elem_id][:2])):
            grid_line = start_point + list(lines_inside_sorted_by_euc[last_elem_id][:2])
        else:
            grid_line = start_point + list(lines_inside_sorted_by_euc[last_elem_id][2:])

        obj_['grid_line_F'] = grid_line

    return filtered_grids_with_text

def extend_short_gridlines(filtered_grids_with_text, svg_height, svg_width):
    '''
    Extend lines that corresponds criteria of being short. This is fix of rounding problem
    (for short lines angle of decl/incl line changes drastically for few degree, which on scale of thousand pixels
    create very big error, that lead to missing lines that creates gridline on cheme
    '''
    min_x = 9999999999
    min_y = 9999999999
    max_x = -1
    max_y = -1
    max_n = max(svg_height, svg_width)
    for obj in filtered_grids_with_text:
        if obj['bbox'][0] < min_x:
            min_x = obj['bbox'][0]
        if obj['bbox'][1] < min_y:
            min_y = obj['bbox'][1]
        if obj['bbox'][2] > max_x:
            max_x = obj['bbox'][2]
        if obj['bbox'][3] > max_y:
            max_y = obj['bbox'][3]

    bbox_all = (min_x, min_y, max_x, max_y)
    num_d_allowed = 2
    for obj in filtered_grids_with_text:
        num_tr = 0
        if euclidean_dist(obj['bbox'][:2], obj['bbox'][2:]) * num_d_allowed > get_line_length(obj['grid_line_F']):
            if euclidean_dist(obj['grid_line_F'][:2], obj['center']) < euclidean_dist(obj['grid_line_F'][2:],
                                                                                      obj['center']):
                start_point = obj['grid_line_F'][:2]
                end_cand = obj['grid_line_F'][2:]
            else:
                start_point = obj['grid_line_F'][2:]
                end_cand = obj['grid_line_F'][:2]

            point_of_intersex = bbox_line_intersection(bbox_all, obj['ext_line'])
            while not point_of_intersex:
                if num_tr > 3:
                    break
                num_tr += 1

                L_ = obj['ext_line']
                bb_center = obj['center']
                var_a_p = extend_line(L_, 40)
                var_a_p = [max(i, 0) for i in var_a_p]
                var_a_p = [min(i, max_n) for i in var_a_p]
                var_a_n = extend_line(L_, -40)
                var_a_n = [max(i, 0) for i in var_a_n]
                var_a_n = [min(i, max_n) for i in var_a_n]
                var_b_p = extend_line_reverse(L_, 40)
                var_b_p = [max(i, 0) for i in var_b_p]
                var_b_p = [min(i, max_n) for i in var_b_p]
                var_b_n = extend_line_reverse(L_, -40)
                var_b_n = [max(i, 0) for i in var_b_n]
                var_b_n = [min(i, max_n) for i in var_b_n]


                # DUMB BUT WORKS, REWRITE BETTER
                max_d_var_a_p = max(euclidean_dist(bb_center, var_a_p[:2]), euclidean_dist(bb_center, var_a_p[2:]))
                max_d_var_a_n = max(euclidean_dist(bb_center, var_a_n[:2]), euclidean_dist(bb_center, var_a_n[2:]))
                max_d_var_b_p = max(euclidean_dist(bb_center, var_b_p[:2]), euclidean_dist(bb_center, var_b_p[2:]))
                max_d_var_b_n = max(euclidean_dist(bb_center, var_b_n[:2]), euclidean_dist(bb_center, var_b_n[2:]))

                if max(max_d_var_a_p, max_d_var_a_n, max_d_var_b_p, max_d_var_b_n) == max_d_var_a_p:
                    ext_line = var_a_p.copy()
                elif max(max_d_var_a_p, max_d_var_a_n, max_d_var_b_p, max_d_var_b_n) == max_d_var_a_n:
                    ext_line = var_a_n.copy()
                elif max(max_d_var_a_p, max_d_var_a_n, max_d_var_b_p, max_d_var_b_n) == max_d_var_b_p:
                    ext_line = var_b_p.copy()
                elif max(max_d_var_a_p, max_d_var_a_n, max_d_var_b_p, max_d_var_b_n) == max_d_var_b_n:
                    ext_line = var_b_n.copy()

                obj['ext_line'] = ext_line.copy()

                point_of_intersex = (bbox_line_intersection(bbox_all, obj['ext_line']) +
                                     bbox_line_intersection(bbox_all, obj['ext_line'][2:] + obj['ext_line'][:2]))


            if num_tr > 3:
                end_point = end_cand
            else:
                end_point = point_of_intersex[0]

            obj['grid_line_F'] = list(start_point) + list(end_point)
    return filtered_grids_with_text





def find_similar_grids(grids):  # Removing duplicated lines
    counter = Counter([i['text'] for i in grids])
    to_check_text = [k for k, v in counter.items() if v > 1]
    to_check = []
    for i in to_check_text:
        tmp = []
        for c, j in enumerate(grids):
            if j['text'] == i:
                tmp.append((c, j))
        to_check.append(tmp)
    return to_check


def group_grids(grids):  # Removing duplicated lines
    def _group_by_diff(to_check):  # Removing duplicated lines
        group = []
        for i in range(len(to_check)):
            for j in range(i + 1, len(to_check)):
                if abs(to_check[i][1]['center'][1] - to_check[j][1]['center'][1]) < 5 \
                        or abs(to_check[i][1]['center'][0] - to_check[j][1]['center'][0]) < 5:
                    group.append((to_check[i][0], to_check[j][0]))
        return group

    # find grids which have the same name
    to_check = find_similar_grids(grids)

    # delete duplicates by rules
    to_del = []
    for i in to_check:
        group = _group_by_diff(i)
        for j in group:
            directions_same = False
            if abs(sum([np.sign(grids[j[0]]['grid_line_F'][0] - grids[j[0]]['grid_line_F'][2])], np.sign(grids[j[1]]['grid_line_F'][0] - grids[j[1]]['grid_line_F'][2]))) != 0: 
                directions_same = True
            if abs(sum([np.sign(grids[j[0]]['grid_line_F'][1] - grids[j[0]]['grid_line_F'][3])], np.sign(grids[j[1]]['grid_line_F'][1] - grids[j[1]]['grid_line_F'][3]))) != 0: 
                directions_same = True
            if not directions_same:
                if len(j) == 2:
                    points = [(grids[j[1]]['grid_line_F'][0], grids[j[1]]['grid_line_F'][1]),
                              (grids[j[1]]['grid_line_F'][2], grids[j[1]]['grid_line_F'][3]),
                              (grids[j[0]]['grid_line_F'][0], grids[j[0]]['grid_line_F'][1]),
                              (grids[j[0]]['grid_line_F'][2], grids[j[0]]['grid_line_F'][3])]
                    grids[j[1]]['grid_line_F'] = [*min(points), *max(points)]
                    to_del.append(j[0])
    grids = [i for c, i in enumerate(grids) if c not in to_del]
    return grids


def filter_grids_duplicates(grids):  # Removing duplicated lines
    # find grids which have the same name
    to_check = find_similar_grids(grids)
    # delete duplicates
    to_del = []
    for i in to_check:
        directions_same = False
        if abs(sum([np.sign(i[0][1]['grid_line_F'][0] - i[0][1]['grid_line_F'][2])], np.sign(i[1][1]['grid_line_F'][0] - i[1][1]['grid_line_F'][2]))) != 0: 
            directions_same = True
        if abs(sum([np.sign(i[0][1]['grid_line_F'][1] - i[0][1]['grid_line_F'][3])], np.sign(i[1][1]['grid_line_F'][1] - i[1][1]['grid_line_F'][3]))) != 0: 
            directions_same = True

        if not directions_same:
        
            to_del_tmp = sorted(i,
                                key=lambda x: euclidean_dist(x[1]['grid_line_F'][:2], x[1]['grid_line_F'][2:]))[:-1]
            to_del_tmp = [i[0] for i in to_del_tmp]
            to_del.extend(to_del_tmp)
    grids = [i for c, i in enumerate(grids) if c not in to_del]
    return grids

def point_closest(line, p):
    x1, y1 = line[:2]
    x2, y2 = line[2:]
    x3, y3 = p
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/det
    return x1+a*dx, y1+a*dy

def extend_line(line,factor = 40): #Extend line free with no SMS or Registration!
    t1=0.5*(1.0+factor)
    x1 = line[0]
    y1 = line[1]
    x2 = int(line[0] +(line[2] - line[0]) * t1)
    y2 = int(line[1] +(line[3] - line[1]) * t1)
    return [x1, y1, x2, y2]

def extend_line_reverse(line,factor = 40): #Extend line free with no SMS or Registration!
    t0=0.5*(1.0-factor)
    x1 = int(line[0] - (line[2] - line[0]) * t0)
    y1 = int(line[1] - (line[3] - line[1]) * t0)
    x2 = line[2]
    y2 = line[3]
    return [x1, y1, x2, y2]

def get_bbox_rectangles_objects(rectangles, rectangles_attributes_dict=None):
    rectangles_objects = []
    for bbox in rectangles:
        rectangles_objects.append(
            {'bbox': dict(zip(['x_min', 'y_min', 'x_max', 'y_max'],
                              fix_coords(bbox))),
             'lines': get_bbox_lines(bbox)})
    if rectangles_attributes_dict:
        for i in rectangles_objects:
            i['color'] = rectangles_attributes_dict[tuple(i['bbox'].values())]['color']

    return rectangles_objects

def filter_objects_by_proportion(closed_objects, thr=0.3, eps=1e-6):
    to_leave = []
    for obj in closed_objects:
        obj_bbox = list(obj['bbox'].values())
        line_lengths = [math.dist(i[:2], i[2:]) for i in get_bbox_lines(obj_bbox)]
        if (min(line_lengths)+eps)/(max(line_lengths)+eps)>=thr:
            to_leave.append(obj)
    return to_leave

def find_boxes_inside(coords):
    boxes_inside_dict = dict([(i, []) for i in range(len(coords))])
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if rectangle_inside_rectangle(coords[i], coords[j]):
                boxes_inside = boxes_inside_dict.get(i, [])
                boxes_inside.append(j)
                boxes_inside_dict[i] = boxes_inside
            elif rectangle_inside_rectangle(coords[j], coords[i]):
                boxes_inside = boxes_inside_dict.get(j, [])
                boxes_inside.append(i)
                boxes_inside_dict[j] = boxes_inside
    return boxes_inside_dict


def get_objects_with_no_insides(closed_objects):
    coords = [list(obj['bbox'].values()) for obj in closed_objects]
    boxes_inside_dict = find_boxes_inside(coords)

    to_return = set()
    for k, v in boxes_inside_dict.items():
        if not v:
            to_return.add(k)
        elif v:
            to_return.update([i for i in v if not boxes_inside_dict[i]])

    return [closed_objects[i] for i in to_return]

def classify_by_exact_text_inside_object(text_inside_object, heuristic,
                                         overall_class='junction box'):
    classified = []
    for text, bbox in text_inside_object:
        if len(text['spans'])==1:
            for class_, pat in heuristic.items():
                text_message = text['spans'][0]['message']
                if pat==text_message:
                    classified.append({'bbox': dict(zip(['x_min','y_min', 'x_max', 'y_max'],bbox)),
                                       'class':overall_class,
                                       'sub_class' : class_})
                    break
    return classified

def classify_by_exact_text(to_check_text, heuristic):
    found_fsds = []
    for i in to_check_text:
        for class_, pat in heuristic.items():
            if i['spans']:
                text_message = i['spans'][0]['message']
                bbox = list(map(int, fix_coords([i['spans'][0]['x0'], i['spans'][0]['y0'], \
                                                 i['spans'][0]['x1'], i['spans'][0]['y1']])))
                if pat==text_message:
                    found_fsds.append({'bbox': dict(zip(['x_min', 'y_min', 'x_max', 'y_max'], \
                                                        bbox)),
                                       'class': 'junction box',
                                       'sub_class': class_})
                    break
    return found_fsds

def get_text_inside_objects(objects, parsed_text,
                            iou_thr=0.8):
    text_inside_objects = {}
    for text in parsed_text:
        bbox = text['x0'], text['y0'], text['x1'], text['y1']
        for obj in objects:
            obj_bbox = list(obj['bbox'].values())
            if iou(obj_bbox, bbox)>iou_thr:
                key = tuple(bbox)
                objects_container = text_inside_objects.get(key, [])
                objects_container.append(tuple(obj_bbox))
                text_inside_objects[key] = objects_container
    return text_inside_objects

def get_text_closest_to_center(text_inside_objects):
    for k, v in text_inside_objects.items():
        if len(v) > 1:
            center_text = center_coord(k)
            arg_min = np.argmin([euclidean_dist(center_text, center_coord(i)) for i in v])
            text_inside_objects[k] = v[arg_min]
        else:
            text_inside_objects[k] = v[0]
    return text_inside_objects

def associate_text_with_info(text_inside_objects, parsed_text):
    text_inside_objects_new = {}
    for k,v in text_inside_objects.items():
        for text in parsed_text:
            bbox = text['x0'], text['y0'], text['x1'], text['y1']
            if tuple(k)==tuple(bbox):
                text_inside = text_inside_objects_new.get(v, [])
                text_inside.append(text)
                text_inside_objects_new[v] = text_inside
    return text_inside_objects_new

def find_possible_switches(text_inside_objects,
                           closed_objects_filtered,
                           disconnected_switch_text=['vfd', 'f', 'v']):
    possible_switches_coords = []
    for k, v in text_inside_objects.items():
        if len(v) == 1:
            text_inside = v[0]
            if len(text_inside['spans']) == 1:
                if text_inside['spans'][0]['message'].lower() in disconnected_switch_text:
                    possible_switches_coords.append(k)

    possible_switches = [i for i in closed_objects_filtered \
                         if (tuple(i['bbox'].values()) not in text_inside_objects.keys()) or \
                         (tuple(i['bbox'].values()) in possible_switches_coords)]
    return possible_switches


def find_disconnected_switches(candidates_no_inner,
                               lines_to_check, lines):
    disconnected_switches = []
    for cand in candidates_no_inner:
        bbox = list(cand['bbox'].values())
        bbox_lines = get_bbox_lines(bbox)

        possible_variants = []
        for bbox_line in bbox_lines:
            line_type2 = check_line_type(bbox_line)
            for line in lines_to_check:
                start, end = line[:2], line[2:]
                line_type1 = check_line_type(line)
                if line_type2 == 'horizontal':
                    if start[1] == bbox_line[1] and bbox_line[2] > start[0] > bbox_line[0] \
                            and line_type1 != line_type2:
                        possible_variants.append((*start, *end))
                        break
                    elif end[1] == bbox_line[1] and bbox_line[2] > end[0] > bbox_line[0] \
                            and line_type1 != line_type2:
                        possible_variants.append((*end, *start))
                        break

                elif line_type2 == 'vertical':
                    if start[0] == bbox_line[0] and bbox_line[3] > start[1] > bbox_line[1] \
                            and line_type1 != line_type2:
                        possible_variants.append((*start, *end))
                        break
                    elif end[0] == bbox_line[0] and bbox_line[3] > end[1] > bbox_line[1] \
                            and line_type1 != line_type2:
                        possible_variants.append((*end, *start)
                                                 )
                        break

        if possible_variants:
            candidate_lines = []
            for line1 in possible_variants:
                line_type1 = check_line_type(line1)
                second_line = []
                for line2 in lines_to_check:
                    line_type2 = check_line_type(line2)
                    if line_type1 != line_type2:
                        if tuple(line2[:2]) == tuple(line1[:2]) or tuple(line2[:2])==tuple(line1[2:]):
                            second_line.append(line2)
                        elif tuple(line2[2:]) == tuple(line1[:2]) or tuple(line2[2:])==tuple(line1[2:]):
                            second_line.append((*line2[2:], *line2[:2]))


                if second_line:
                    if len(second_line) == 1:
                        candidate_lines.append([line1, second_line[0]])

            if candidate_lines:
                for first_line, second_line in candidate_lines:
                    start1, end1 = first_line[:2], first_line[2:]
                    start2, end2 = second_line[:2], second_line[2:]
                    flag = []
                    for line2 in lines:
                        if not ((line2[:2] == start1 and line2[2:] == end1) \
                                or (line2[:2] == end1 and line2[2:] == start1)) and \
                                not ((line2[:2] == start2 and line2[2:] == end2) \
                                     or (line2[:2] == end2 and line2[2:] == start2)):

                            if check_line_type(line2) in ('horizontal',
                                                          'vertical'):
                                rule = not ((end2[0] == line2[0] and line2[3] >= end2[1] >= line2[1]) or \
                                            (end2[1] == line2[1] and line2[2] >= end2[0] >= line2[0]))

                                flag.append(rule)


                    if all(flag):
                        disconnected_switches.append([cand, [first_line, second_line]])

    return disconnected_switches


def filter_disconnected_switches(disconnected_switches,
                                 candidates_no_inner,
                                 lines_to_check,
                                 max_area=8000,
                                 max_bbox_ls_trigger_l_ratio=0.8,
                                 min_bbox_ls_trigger_l_ratio=0.1,
                                 min_max_trigger_ls_ratio=0.1):
    to_del = []
    cand_bboxes = [tuple(i['bbox'].values()) for i in candidates_no_inner]
    for c, i in enumerate(disconnected_switches):
        cand, trigger_lines = i
        cand_lines = cand['lines']
        bbox_lines = get_bbox_lines(list(cand['bbox'].values()))
        cand_lines += bbox_lines

        # intersection check
        len_intersect = len(set(map(tuple, cand_lines)). \
                            intersection(set(map(fix_coords_line, trigger_lines))))
        if len_intersect:
            to_del.append(c)
            continue

        # trigger lines check
        length_bbox_lines = np.array([abs(i[0] - i[2]) if i[1] == i[3] else abs(i[1] - i[3]) \
                                      for i in bbox_lines])
        length_trigger_lines = np.array([abs(i[0] - i[2]) if i[1] == i[3] else abs(i[1] - i[3]) \
                                         for i in trigger_lines])
        flag = False
        for l in length_trigger_lines:
            if not l or ((length_bbox_lines / l) <= max_bbox_ls_trigger_l_ratio).any() or ((l / length_bbox_lines) <= min_bbox_ls_trigger_l_ratio).any():
                to_del.append(c)
                flag = True
                break

        if flag:
            continue

        # check if first line is intersected by any other
        l = trigger_lines[0]
        l_linestring = shapely.geometry.LineString([l[:2], l[2:]])
        flag = False
        for line2 in lines_to_check:
            line2_linestring = LineString([line2[:2], line2[2:]])
            intersection = l_linestring.intersection(line2_linestring)
            if intersection and isinstance(intersection, shapely.geometry.Point):
                coords = tuple(map(int, intersection.coords[0]))
                if not (tuple(l[:2]) == coords or tuple(l[2:]) == coords):
                    to_del.append(c)
                    break

        if flag:
            continue

        # check if second line is intersected by other one
        l = trigger_lines[1]
        l_linestring = shapely.geometry.LineString([l[:2], l[2:]])
        flag = False
        for line2 in lines_to_check:
            line2_linestring = LineString([line2[:2], line2[2:]])
            intersection = l_linestring.intersection(line2_linestring)
            if intersection and isinstance(intersection, shapely.geometry.Point):
                coords = tuple(map(int, intersection.coords[0]))
                if (tuple(l[:2]) == coords or tuple(l[2:]) == coords) and \
                        (coords != tuple(trigger_lines[0][:2]) and coords != tuple(trigger_lines[0][2:])):
                    to_del.append(c)
                    flag = True
                    break
        if flag:
            continue

        # check if ratio between lines length is too big
        if min(length_trigger_lines) / max(length_trigger_lines) < min_max_trigger_ls_ratio:
            to_del.append(c)

        # line inside other object check
        bboxes_to_comp = set(cand_bboxes).difference(tuple(cand['bbox'].values()))
        for l in trigger_lines:
            flag = False
            for bbox in bboxes_to_comp:
                if is_point_inside_rect(l[:2], bbox) and is_point_inside_rect(l[2:], bbox):
                    to_del.append(c)
                    flag = True
                    break
            if flag:
                break

        # # check area
        cand_area = (cand['bbox']['x_max'] - cand['bbox']['x_min']) * (cand['bbox']['y_max'] - cand['bbox']['y_min'])
        if cand_area > max_area:
            to_del.append(c)
    disconnected_switches = [i for c, i in enumerate(disconnected_switches)
                             if c not in to_del]
    return disconnected_switches

def form_bbox_switch(candidates, disconnected_switches, to_return_lines=False):
    new_disconnected_switches = []
    for i in disconnected_switches:
        cand, trigger_lines = i
        try:
            candidates.remove(cand)
        except:
            pass
        xs = list(chain(*[[i[0], i[2]] for i in trigger_lines]))
        ys = list(chain(*[[i[1], i[3]] for i in trigger_lines]))
        x_min, y_min, x_max, y_max = cand['bbox'].values()
        x_min = min(min(xs), x_min)
        x_max = max(max(xs), x_max)
        y_min = min(min(ys), y_min)
        y_max = max(max(ys), y_max)
        disconnected_switch = {'bbox': {'x_min': x_min - 1, 'y_min': y_min - 1, 'x_max': x_max + 1, 'y_max': y_max + 1},
                               'class' : 'junction box',
                               'sub_class': 'disconnected switch'
                               }
        if to_return_lines:
            disconnected_switch['lines'] = trigger_lines
        candidates.append(disconnected_switch)
        new_disconnected_switches.append(disconnected_switch)

    return candidates, new_disconnected_switches


def is_parallelogram(lines):
    def are_lines_parallel(line1, line2):
        # Check if two lines represented by line segments are parallel
        dx1 = line1[0] - line1[2]
        dy1 = line1[1] - line1[3]
        dx2 = line2[0] - line2[2]
        dy2 = line2[1] - line2[3]

        # Check if the cross product of the direction vectors is zero
        return dx1 * dy2 == dx2 * dy1

    if len(lines) != 4:
        return False  # A parallelogram must have four sides


    combs = list(combinations(lines, 2))
    # Check if opposite sides are parallel and have equal length
    return any([get_line_length(i[0])==get_line_length(i[1]) \
                and are_lines_parallel(i[0], i[1]) \
                for i in combs])

def most_frequent(list_color_cand):
    '''Return most frequent color, from list of colors'''
    unique, counts = np.unique(list_color_cand, return_counts=True)
    index = np.argmax(counts)
    return unique[index]


def key_plan_line_distance(bbox, line):
    bbox_side = [bbox[0], bbox[3], bbox[2], bbox[3]]
    line_type = check_line_type(line)
    x1, y1, x2, y2 = line
    x3, y3, x4, y4 = bbox_side
    dist = float('inf')
    if line_type == 'horizontal' and line_type == check_line_type(bbox_side):
        if x2 >= x3 >= x1 or x4 >= x1 >= x3 or x2 >= x4 >= x1 or x4 >= x2 >= x3:
            dist = abs(y1 - y3)

    return dist

def filter_key_plan_grids(grids, parsed_text,
                      lines,
                      svg_width,
                      svg_height,
                      area_ratio=0.9,
                      max_dist_to_text=0.3,
                      max_dist_to_line=1):
    to_del = []

    key_plan_coords = None
    for i in parsed_text:
        message = ''.join([j['message'] for j in i['spans']])
        if 'key' in message.lower() and 'plan' in message.lower():
            key_plan_coords = [i['x0'], i['y0'], i['x1'], i['y1']]
            break

    if key_plan_coords:
        lines_to_check = [i for i in lines \
                          if check_line_type(i) == 'horizontal'
                          and i[0] != i[2]]

        distance_metric = partial(pairwise_distances,
                                  metric=key_plan_line_distance)
        distance_matrix = distance_metric(X=[key_plan_coords], Y=lines_to_check)
        min_distance = distance_matrix.min()

        if min_distance <= max_dist_to_line:
            arg_min = np.argmin(distance_matrix)
            key_plan_line = lines_to_check[arg_min]
        else:
            key_plan_line = None

        if not key_plan_line:
            areas = [abs(i['bbox'][2] - i['bbox'][0]) * abs(i['bbox'][3] - i['bbox'][1]) \
                     for i in grids]
            indices = np.where(np.array(areas) / max(areas) < area_ratio)[0]
            to_check = np.array([grids[i]['bbox'] for i in indices])
            if to_check.size:
                to_check = to_check.astype(float)
                to_check[:, 0] = to_check[:, 0] / svg_width
                to_check[:, 2] = to_check[:, 2] / svg_width
                to_check[:, 3] = to_check[:, 3] / svg_height
                to_check[:, 1] = to_check[:, 1] / svg_height
                key_plan_coords = [key_plan_coords[0] / svg_width, key_plan_coords[1] / svg_height,
                                   key_plan_coords[2] / svg_width, key_plan_coords[3] / svg_height]
                to_del = indices[(euclidean_distances(to_check, [key_plan_coords]) < max_dist_to_text).reshape(-1)]
        else:
            grids_centers = np.array([[(i['bbox'][2] + i['bbox'][0]) // 2, (i['bbox'][3] + i['bbox'][1]) // 2] \
                                  for i in grids])

            to_del = np.where(np.all([grids_centers[:, 0] >= key_plan_line[0], grids_centers[:, 0] <= key_plan_line[2],
                                  grids_centers[:, 1] > key_plan_line[1]],
                                 axis=0))[0]
    grids = [i for c,i in enumerate(grids) if c not in to_del]
    return grids


def delete_objects_inside_object(objects1, objects2):
    to_del = []
    for object2 in objects2:
        bbox2 = list(object2['bbox'].values())
        for object1 in objects1:
            bbox1 = list(object1['bbox'].values())
            if rectangle_inside_rectangle(bbox1, bbox2):
                to_del.append(object2)
                break
    return [i for i in objects2 if i not in to_del]

