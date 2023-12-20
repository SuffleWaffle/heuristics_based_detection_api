import numpy as np

from src_logging.log_config import setup_logger
from src_utils.geometry_utils import bb_intersection_over_union

logger = setup_logger(__name__)


def merge_and_NMS(result_detection_obj):
    """
    The merge_and_NMS function takes in a dictionary of detected objects and returns the same dictionary with
        non-maximal suppression applied.

    :param result_detection_obj: Store the result of both detections
    :return: A dictionary with the same as input structure
    """
    list_obj_ = []
    list_bb_NMS = []
    object_sorted_list = sorted(result_detection_obj['data'], key=lambda d: d['confidence'], reverse=True)
    for obj_ in object_sorted_list:
        x0 = obj_['bbox']['x_min']
        y0 = obj_['bbox']['y_min']
        x1 = obj_['bbox']['x_max']
        y1 = obj_['bbox']['y_max']
        box = np.array([x0, y0, x1, y1])
        b = box.astype(int)
        list_of_iou = [bb_intersection_over_union(b, i) for i in list_bb_NMS]
        if list_of_iou.__len__() > 1:
            if max(list_of_iou) <= 0.5:
                list_bb_NMS.append(list(b))
                list_obj_.append(obj_)
        else:
            list_bb_NMS.append(list(b))
            list_obj_.append(obj_)
    result_detection_obj['data'] = list_obj_
    return result_detection_obj
