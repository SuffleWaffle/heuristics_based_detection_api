from copy import deepcopy
from itertools import combinations

import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

def center_coord(coord):
    x = int((coord[0] + coord[2]) // 2)
    y = int((coord[1] + coord[3]) // 2)
    return x, y
def is_point_inside_bbox(point, bbox):
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax


def get_n_closest(origins, candidates, dist_metric=euclidean_distances, n=50,
                  thr=None, return_indices=False):
    def _filter_by_threshold(sorted_args, sorted_distances, thr):
        args = []
        for i, dist in enumerate(sorted_distances):
            args.append(sorted_args[i][np.where(dist <= thr)[0]])
        return args

    if not isinstance(origins, list):
        origins = [origins]
    if not isinstance(candidates, np.ndarray) and candidates is None:
        candidates = origins

    distances = dist_metric(origins, candidates)
    sorted_args = np.argsort(distances)
    sorted_distances = np.sort(distances)

    if not (thr is None):
        sorted_args = _filter_by_threshold(sorted_args, sorted_distances, thr)

    if not (n is None):
        if isinstance(sorted_args, np.ndarray):
            sorted_args = sorted_args[:, :n]
        else:
            sorted_args = [i[:n] for i in sorted_args]

    if not return_indices:
        if ((isinstance(candidates, np.ndarray) and candidates.size) or (
                isinstance(candidates, (list, tuple)) and candidates)) and \
                ((isinstance(sorted_args, np.ndarray) and sorted_args.size)
                 or (isinstance(sorted_args, (list, tuple)) and sorted_args)):
            return [[candidates[i] for i in j] for j in sorted_args]
        else:
            return []
    else:
        if ((isinstance(candidates, np.ndarray) and candidates.size)
            or (isinstance(candidates, (list, tuple)) and candidates)) and \
                ((isinstance(sorted_args, np.ndarray) and sorted_args.size)
                 or (isinstance(sorted_args, (list, tuple)) and sorted_args)):
            return [[candidates[i] for i in j] for j in sorted_args], sorted_args
        else:
            return [], []


def fix_coords_line(line):
    start, end = line[:2], line[2:]
    line = sorted([start, end])
    line = [*line[0], *line[1]]
    return tuple(line)

def fix_coords(tags_coords):
    x1, y1, x2, y2 = tags_coords
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)
    return xmin, ymin, xmax, ymax


def is_part_of_other(line1, line2, types,
                     rads1=None, rads2=None, angle_tol=0.1,
                     line_eq_tol=2):
    types = list(set(types))

    if len(types) == 1 and types[0] == 'vertical':
        if line1[0] == line2[0]:
            if line2[3] >= line1[1] >= line2[1] or (line1[1] <= line2[1] and line1[3] >= line2[3]):
                return True
            else:
                return False
        else:
            return False

    elif len(types) == 1 and types[0] == 'horizontal':
        if line1[1] == line2[1]:
            if line2[2] >= line1[0] >= line2[0] or (line1[0] <= line2[0] and line1[2] >= line2[2]):
                return True
            else:
                return False
        else:
            return False

    elif rads1 or rads2:
        m, b = compute_slope_intercept(line2)
        line2_x_max, line2_x_min = max([line2[2], line2[0]]), min([line2[2], line2[0]])
        line2_y_max, line2_y_min = max([line2[1], line2[3]]), min([line2[1], line2[3]])
        if (abs(m * line1[0] + b - line1[1]) < line_eq_tol or abs(m * line1[2] + b - line1[3]) < line_eq_tol) and \
                ((line2_x_max >= line1[0] >= line2_x_min and line2_y_max >= line1[1] >= line2_y_min) or \
                 (line2_x_max >= line1[2] >= line2_x_min and line2_y_max >= line1[3] >= line2_y_min)):

            if (rads1 and rads2 and (abs(rads1[0] - rads2[0]) < angle_tol)) or \
                    not (rads1 and rads2):
                return True
            else:
                return False
        else:
            return False


def check_line_type(line):
    if line[1] == line[3] and not line[0] == line[2]:
        return 'horizontal'
    elif line[0] == line[2] and not line[1] == line[3]:
        return 'vertical'
    else:
        return 'other'

def get_line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def bbox_line_intersection(bbox, line, tol_=0):
    # TODO no handle if line[0]>line[2]
    bbox_lines = get_bbox_lines(bbox)
    points_intersect = []
    for i in bbox_lines:
        if line[0] == line[2] or line[1] == line[3]:
            flag, point = line_v_h_intersection(line, i, tol_)
        else:
            m, b = compute_slope_intercept(line)
            flag, point = line_v_h_o_intersection(i, [*line, m, b], tol_)
        if flag:
            points_intersect.append(point)

    return points_intersect


def is_point_on_line(point, line, tol=0):
    x0, y0 = point
    x1, y1, x2, y2 = line
    # Check if the line is vertical
    if x1 == x2:
        return abs(x0 - x1) <= tol and (y0 >= min(y1, y2)) and (y0 <= max(y1, y2))
    # Check if the line is horizontal
    elif y1 == y2:
        return abs(y0 - y1) <= tol and (x0 >= min(x1, x2)) and (x0 <= max(x1, x2))
    # If the line is neither vertical nor horizontal, return False
    else:
        return (x0 >= min(x1, x2)) and (x0 <= max(x1, x2)) \
            and (y0 >= min(y1, y2)) and (y0 <= max(y1, y2))


def get_bbox_lines(bbox):
    bbox_lines = [
        [bbox[0], bbox[1], bbox[0], bbox[3]],
        [bbox[2], bbox[1], bbox[2], bbox[3]],
        [bbox[0], bbox[1], bbox[2], bbox[1]],
        [bbox[0], bbox[3], bbox[2], bbox[3]]
    ]
    return bbox_lines


def line_intersection(line1, line2):
    if (line1[0] == line1[2] or line1[1] == line1[3]) and \
            (line2[0] == line2[2] or line2[1] == line2[3]):
        return line_v_h_intersection(line1, line2, 0)

    elif (line1[0] == line1[2] or line1[1] == line1[3]) and \
            not (line2[0] == line2[2] or line2[1] == line2[3]):
        m, b = compute_slope_intercept(line2)
        return line_v_h_o_intersection(line1, [*line2, m, b], 0)

    elif (line2[0] == line2[2] or line2[1] == line2[3]) and \
            not (line1[0] == line1[2] or line1[1] == line1[3]):
        m, b = compute_slope_intercept(line1)
        return line_v_h_o_intersection(line2, [*line1, m, b], 0)

    else:
        m1, b1 = compute_slope_intercept(line1)
        m2, b2 = compute_slope_intercept(line2)
        return line_o_o_intersection([*line1, m1, b1], [*line2, m2, b2], 0)


def line_v_h_o_intersection(line1, line2, tol1=1, tol2=0,
                            for_sld=False):
    # line1 is assumed to be a vertical/horizontal line
    # line2 is assumed to be an inclined line
    x1, y1, x2, y2= line1
    x3, y3, x4, y4, m2, b2 = line2

    if x1 == x2:  # line1 is a vertical line
        x_int = x1
        y_int = m2 * x_int + b2
    elif y1 == y2:  # line1 is a horizontal line
        y_int = y1
        x_int = (y_int - b2) / m2

    # check if intersection point is within the bounds of both lines
    x_int = round(x_int)
    y_int = round(y_int)

    min_x1, max_x1 = min(x1, x2), max(x1, x2)
    min_y1, max_y1 = min(y1, y2), max(y1, y2)

    min_x2, max_x2 = min(x3, x4), max(x3, x4)
    min_y2, max_y2 = min(y3, y4), max(y3, y4)

    if (
        min_x1 - tol1 <= x_int <= max_x1 + tol1
        and min_x2 - tol1 <= x_int <= max_x2 + tol1
        and min_y1 - tol1 <= y_int <= max_y1 + tol1
        and min_y2 - tol1 <= y_int <= max_y2 + tol1
    ):
        if for_sld:
            if abs(x_int - x1) <= tol2 and abs(y_int - y1) <= tol2:
                return True, (x1, y1)
            elif abs(x_int - x2) <= tol2 and abs(y_int - y2) <= tol2:
                return True, (x2, y2)
            else:
                return False, None
        else:
            return True, (x_int, y_int)

    else:
        return False, None

def lines_intersection_distance_vectorized(X, Y=None):
    if not Y:
        X = list(map(fix_coords, X))
        X = np.array(X)
        subtract_x = np.subtract.outer(X[:, 2], X[:, 0])
        subtract_y = np.subtract.outer(X[:, 1], X[:, 3])
        result = np.any([subtract_x < 0, subtract_y < 0], axis=1).astype(int)
    else:
        X = list(map(fix_coords, X))
        Y = list(map(fix_coords, Y))
        X = np.array(X)
        Y = np.array(Y)
        subtract_x1 = np.subtract.outer(X[:, 2], Y[:, 0])
        subtract_x2 = np.negative(np.subtract.outer(X[:, 0], Y[:, 2]))
        subtract_y1 = np.subtract.outer(X[:, 3], Y[:, 1])
        subtract_y2 = np.negative(np.subtract.outer(X[:, 1], Y[:, 3]))
        result = np.any([subtract_x1 < 0, subtract_y1 < 0,
                         subtract_y2 < 0, subtract_x2 < 0], axis=0).astype(int)

    return np.array(result)

def get_lines_by_intersection_dict(lines, points_intersection):
    lines_with_intersections = []
    for k, v in points_intersection.items():
        start_point, end_point = lines[k][:2], lines[k][2:]
        points = list(v) + [start_point, end_point]
        possible_lines = [fix_coords_line(tuple([*i[0], *i[1]])) for i in list(combinations(points, 2))]
        lines_with_intersections.extend(possible_lines)
    return lines_with_intersections

def split_all_lines_by_intersections(lines, tol1=1, tol2=0,
                                     for_sld=False):
    original_lines = deepcopy(lines)
    # get different types of lines
    horizontal = []
    vertical = []
    other = []
    for c, i in enumerate(lines):
        if check_line_type(i) == 'horizontal':
            horizontal.append((i, c))
        elif check_line_type(i) == 'vertical':
            vertical.append((i, c))
        else:
            other.append((i, c))
    # find slopes and intercepts
    slopes_intercepts = {}
    for line, idx in other:
        slopes_intercepts[idx] = compute_slope_intercept(line)

    points_intersection_other = {}
    points_intersection_horizontal = {}
    points_intersection_vertical = {}
    # precompute 1
    _, n_closest_args = get_n_closest([i[0] for i in horizontal],
                                      [i[0] for i in other], dist_metric=lines_intersection_distance_vectorized, thr=0,
                                      n=None,
                                      return_indices=True)

    for c, (horizontal_line, idx1) in enumerate(horizontal):
        for arg in n_closest_args[c]:
            other_line, idx2 = other[arg]
            flag, point = line_v_h_o_intersection(horizontal_line, [*other_line, *slopes_intercepts[idx2]],
                                                  tol1=tol1, tol2=tol2,
                                                  for_sld=for_sld)
            if point:
                set_points = points_intersection_other.get(idx2, set())
                set_points.add(tuple(point))
                points_intersection_other[idx2] = set_points

    # precompute 2
    _, n_closest_args = get_n_closest([i[0] for i in vertical],
                                      [i[0] for i in other], dist_metric=lines_intersection_distance_vectorized, thr=0,
                                      n=None,
                                      return_indices=True)
    for c, (vertical_line, idx1) in enumerate(vertical):
        for arg in n_closest_args[c]:
            other_line, idx2 = other[arg]
            flag, point = line_v_h_o_intersection(vertical_line, [*other_line, *slopes_intercepts[idx2]],
                                                  for_sld=for_sld)
            if point:
                set_points = points_intersection_other.get(idx2, set())
                set_points.add(tuple(point))
                points_intersection_other[idx2] = set_points

    # precompute 3
    _, n_closest_args = get_n_closest([i[0] for i in horizontal],
                                      [i[0] for i in vertical], dist_metric=lines_intersection_distance_vectorized,
                                      thr=0, n=None,
                                      return_indices=True)
    for c, (horizontal_line, idx1) in enumerate(horizontal):
        for arg in n_closest_args[c]:
            vertical_line, idx2 = vertical[arg]
            flag, point = line_v_h_intersection_special(horizontal_line, vertical_line)
            if point is not None:
                # horizontal set of points
                set_points = points_intersection_horizontal.get(idx1, set())
                set_points.add(tuple(point))
                points_intersection_horizontal[idx1] = set_points
                # vertical set of points
                set_points = points_intersection_vertical.get(idx2, set())
                set_points.add(tuple(point))
                points_intersection_vertical[idx2] = set_points

    # get everything togather
    lines_with_intersections = get_lines_by_intersection_dict(lines,
                                                              points_intersection_horizontal)
    lines_with_intersections.extend(get_lines_by_intersection_dict(lines,
                                                              points_intersection_vertical))
    points_intersection_other = dict([(k, v) for k, v in points_intersection_other.items() if len(v) < 3])
    lines_with_intersections.extend(get_lines_by_intersection_dict(lines,
                                                              points_intersection_other))
    lines_with_intersections = list(set(lines_with_intersections))
    lines = [i for c, i in enumerate(lines) \
             if c not in \
             list(points_intersection_horizontal.keys()) + list(points_intersection_vertical.keys()) + \
             list(points_intersection_other.keys())] + \
            lines_with_intersections

    lines = list(set([tuple(map(int, i)) for i in lines]))
    return original_lines, lines

def line_v_h_intersection_special(horizontal_line, vertical_line):
    check_vh_intersect = lambda x, y: (x[0] <= y[0] <= x[2] or x[0] <= y[0] <= x[2]) \
                                      and (y[1] <= x[1] <= y[3] or y[1] <= x[1] <= y[3])

    if_intersect, point = check_vh_intersect(horizontal_line, vertical_line), \
        [vertical_line[0], horizontal_line[1]]

    min_len = min(abs(horizontal_line[0] - point[0]), abs(horizontal_line[2] - point[0]))
    denominator = abs(horizontal_line[0] - horizontal_line[2])
    flag = True
    if denominator:
        flag = min_len / abs(horizontal_line[0] - horizontal_line[2]) < 0.1

    if (tuple(vertical_line[2:]) == tuple(point) or tuple(vertical_line[2:]) == tuple(point)) \
            and if_intersect and flag:
        return True, point
    else:
        return False, None

def line_v_h_intersection(line1, line2, tol=1):
    check_vh_intersect = lambda x, y: (x[0] <= y[0] - tol <= x[2] or x[0] <= y[0] + tol <= x[2]) \
                                      and (y[1] <= x[1] - tol <= y[3] or y[1] <= x[1] + tol <= y[3])
    if line1 == line2:
        return True, [line1[0], line1[1]]

    elif line1[1] == line1[3] and line2[0] == line2[2]:
        return check_vh_intersect(line1, line2), [line2[0], line1[1]]

    elif line1[0] == line1[2] and line2[1] == line2[3]:
        return check_vh_intersect(line2, line1), [line1[0], line2[1]]

    return False, None


def compute_slope_intercept(line):
    x1, y1, x2, y2 = line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def line_o_o_intersection(line1, line2, tol1=1):
    # line1 is assumed to be a vertical/horizontal line
    # line2 is assumed to be an inclined line
    x1, y1, x2, y2, m1, b1 = line1
    x3, y3, x4, y4, m2, b2 = line2
    if m1 == m2:
        return False, None
    x_int = (b2 - b1) / (m1 - m2)
    y_int = m1 * x_int + b1

    # check if intersection point is within the bounds of both lines
    x_int = round(x_int)
    y_int = round(y_int)

    min_x1, max_x1 = min(x1, x2), max(x1, x2)
    min_y1, max_y1 = min(y1, y2), max(y1, y2)

    min_x2, max_x2 = min(x3, x4), max(x3, x4)
    min_y2, max_y2 = min(y3, y4), max(y3, y4)

    if (
            min_x1 - tol1 <= x_int <= max_x1 + tol1
            and min_x2 - tol1 <= x_int <= max_x2 + tol1
            and min_y1 - tol1 <= y_int <= max_y1 + tol1
            and min_y2 - tol1 <= y_int <= max_y2 + tol1
    ):

        return True, (x_int, y_int)

    else:
        return False, None

def euclidean_dist(point1, point2):
    x0, y0 = point1
    x1, y1 = point2

    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

def bb_intersection_over_union(boxA: np.array,
                               boxB: np.array):
    # Calculate the intersection over union of two bounding boxes.
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def rectangle_inside_rectangle(table_coords1, table_coords2, c=0):
    min_x1 = min(table_coords1[0], table_coords1[2]) - c
    max_x1 = max(table_coords1[0], table_coords1[2]) + c
    min_y1 = min(table_coords1[1], table_coords1[3]) - c
    max_y1 = max(table_coords1[1], table_coords1[3]) + c

    min_x2 = min(table_coords2[0], table_coords2[2])
    max_x2 = max(table_coords2[0], table_coords2[2])
    min_y2 = min(table_coords2[1], table_coords2[3])
    max_y2 = max(table_coords2[1], table_coords2[3])

    return min_x2 > min_x1 and min_y2 > min_y1 and max_x2 < max_x1 and max_y2 < max_y1

def find_vect_direct_rads(vector):
    degrees = np.degrees(np.arctan(np.sqrt(((vector[1] - vector[3]) ** 2) / ((vector[0] - vector[2]) ** 2))))

    if vector[0] - vector[2] == 0 and vector[1] < vector[3]:
        degrees = 90

    if vector[0] - vector[2] == 0 and vector[1] > vector[3]:
        degrees = -90

    if vector[0] < vector[2] and vector[1] > vector[3]:
        degrees = -degrees

    if vector[0] > vector[2] and vector[1] < vector[3]:
        degrees = 180 - degrees

    if vector[0] > vector[2] and vector[1] > vector[3]:
        degrees = 180 + degrees

    return np.deg2rad(degrees)

def create_vector_in_direction(start, radians, radius
                               ):
    end_point = rotate(start, (start[0] + radius, start[1]), radians)
    return end_point

def iou(box1, box2):
    if rectangle_inside_rectangle(box1, box2, 5):
        return 1
    # determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # calculate the area of the intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # calculate the intersection over union
    iou_score = intersection_area / union_area

    return iou_score

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def is_line_in_bbox(line, bbox):
    x1, y1, x2, y2 = line
    x_min, y_min, x_max, y_max = bbox
    # check if both endpoints of the line are inside the bbox
    if x_min <= x1 <= x_max and y_min <= y1 <= y_max and \
            x_min <= x2 <= x_max and y_min <= y2 <= y_max:
        return True
    # check if the line intersects any of the bbox edges
    if x1 < x_min and x2 < x_min:
        return False
    if x1 > x_max and x2 > x_max:
        return False
    if y1 < y_min and y2 < y_min:
        return False
    if y1 > y_max and y2 > y_max:
        return False
    # check if the line intersects the left or right bbox edge
    if x1 < x_min or x2 < x_min or x1 > x_max or x2 > x_max:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y_left = m * x_min + b
        y_right = m * x_max + b
        if y_min <= y_left <= y_max or y_min <= y_right <= y_max:
            return True
        else:
            return False
    # check if the line intersects the top or bottom bbox edge
    if y1 < y_min or y2 < y_min or y1 > y_max or y2 > y_max:
        m = (x2 - x1) / (y2 - y1)
        b = x1 - m * y1
        x_top = m * y_min + b
        x_bottom = m * y_max + b
        if x_min <= x_top <= x_max or x_min <= x_bottom <= x_max:
            return True
        else:
            return False
    # the line does not intersect any bbox edge, return False
    return False

def scale(line, original_size, new_size):
    x0, y0, x1, y1 = line

    Rx = new_size[0] / original_size[0]
    Ry = new_size[1] / original_size[1]

    return np.round(x0 * Rx), np.round(y0 * Ry), np.round(x1 * Rx), np.round(y1 * Ry)

def center_start_end_point_dist(lines, candidates):
    centers = [center_coord(i) for i in lines]
    candidates_starts, candidates_ends = [i[:2] for i in candidates], \
                                         [i[2:] for i in candidates]
    return np.minimum(euclidean_distances(centers, candidates_starts),
                      euclidean_distances(centers, candidates_ends))


def is_point_inside_rect(point, rect):
    """Check if a point is inside a rectangle"""
    x, y = point
    x0, y0, x1, y1 = rect
    return x0 <= x < x1 and y0 <= y < y1

