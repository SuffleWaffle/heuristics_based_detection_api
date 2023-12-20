from tqdm import tqdm

from src_utils.geometry_utils import get_line_length, center_coord, get_n_closest,\
    is_point_inside_bbox


def filter_by_color_nin(objects, attributes,
                              colors=[None, '#FFFFFF']):
    attributes = [i.get('color', None) for i in attributes]
    to_leave = []
    for c, object in enumerate(objects):
        if attributes[c] not in colors:
            to_leave.append(object)

    return to_leave

def filter_by_color_in(objects, attributes, colors=['#000000']):

    attributes = [i.get('color', None) for i in attributes]
    to_leave = []
    for c, object in enumerate(objects):
        if attributes[c] in colors:
            to_leave.append(object)

    return to_leave

def del_full_lines_inside_objects(objects, lines,
                                  n=100):
    centers_obj = [center_coord(obj) for obj in objects]
    centers_lines = [center_coord(line) for line in lines]
    _, n_closest_idx = get_n_closest(centers_obj, centers_lines, n=n,
                                     return_indices=True)

    to_del = []
    for c, obj in enumerate(objects):
        for idx in n_closest_idx[c]:
            if is_point_inside_bbox(lines[idx][:2], obj) and is_point_inside_bbox(lines[idx][2:], obj):
                to_del.append(lines[idx])

    return to_del

def filter_lines_by_length(lines, svg_width, svg_height, thr=0.8):
    new_lines = []
    for line in lines:
        if abs(line[0] - line[2]) < thr*svg_width and abs(line[1] - line[3]) <thr*svg_height \
        and get_line_length(line)>2:
            new_lines.append(line)
    return new_lines

def generate_possible_variants_lines(points, size_y, size_x):
    min_x, max_x, min_y, max_y = points
    variants = [[min_x - 1, max_x - 1, min_y - 1, max_y - 1],
                [min_x + 1, max_x + 1, min_y + 1, max_y + 1],

                [min_x, max_x, min_y + 1, max_y + 1],
                [min_x, max_x, min_y - 1, max_y - 1],

                [min_x + 1, max_x + 1, min_y - 1, max_y - 1],
                [min_x - 1, max_x - 1, min_y + 1, max_y + 1],

                [min_x - 1, max_x - 1, min_y, max_y],
                [min_x + 1, max_x + 1, min_y, max_y],

                [min_x, max_x, min_y, max_y]
                ]

    return [i for i in variants if i[0] > 0 and i[0] < size_x and i[1] > 0 and i[1] < size_x \
            and 0 < i[2] < size_y and 0 < i[2] < size_y]

def del_by_existence_objects(img_processed, objects, threshold_line=0.103,
                            threshold_obj=0.5):
    to_leave = []
    for object_ in objects:
        object_lines = object_['lines']
        lines_to_del = del_by_existence_lines(img_processed, object_lines,
                               threshold_line)
        if not len(lines_to_del)/len(object_lines)>threshold_obj:
            to_leave.append(object_)
    return to_leave
def del_by_existence_lines(img_processed, lines, threshold=0.103):
    to_del = []
    for line in tqdm(lines):
        min_x, max_x = min(line[0], line[2]), max(line[0], line[2])
        min_y, max_y = min(line[1], line[3]), max(line[1], line[3])
        variants = generate_possible_variants_lines([min_x, max_x, min_y, max_y],
                                                    size_x=img_processed.shape[1],
                                                    size_y=img_processed.shape[0])
        try:
            if min_x == max_x and min_y != max_y:
                val = max([(img_processed[min_y:max_y, min_x] != 255).mean() \
                           for min_x, max_x, min_y, max_y in variants])
            elif min_y == max_y and min_x != max_x:
                val = max([(img_processed[min_y, min_x:max_x] != 255).mean() \
                           for min_x, max_x, min_y, max_y in variants])
            elif min_y != max_y and min_x != max_x:
                val = max([(img_processed[min_y:max_y, min_x:max_x] != 255).mean() \
                           for min_x, max_x, min_y, max_y in variants])
            else:
                val = 1
        except:
            val = 1

        if val < threshold:
            to_del.append(line)
    return to_del