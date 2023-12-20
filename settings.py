import os

# ________________________________________________________________________________
# --- APPLICATION SETTINGS ---
# PARALLEL_PROC_TIMEOUT: int = int(os.getenv("PARALLEL_PROC_TIMEOUT"))

AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY")
AWS_REGION_NAME: str = os.getenv("AWS_REGION_NAME")


# ________________________________________________________________________________
# --- ALGORITHM SETTINGS ---
GRID_LINES_CONF = dict(
    get_close_objects_based_on_groups=dict(tol=5),
    filter_duplicate_bboxes=dict(area_ratio_thr=0.95),
    filter_grids=dict(
        regexes=[r'^[a-zA-Z]{2}$', r'^[a-zA-Z][0-9]$', r'^[a-zA-Z]{1}$', r'^[A-Z]\.\d+$', r'^[a-zA-Z][0-9][0-9]$',
                 r'^\d+\.[A-Za-z]$', r'^\d+[A-Za-z]?$', r'^[A-Za-z]\d+\.\d+$', r'^[A-Za-z]\s\d+$', r'^[A-Za-z]\s[A-Za-z]+$',
                 r'^[A-Za-z]{1,2}\d+\.\d+$', r'^\d+\.\d+\.\d+$', r'^[A-Z]{1,2}-\d+(\.\d+)?$', r'^\d+[A-Za-z]$',
                 r'^[A-Za-z]\.\d+\.\d+$', r'^[A-Z]{1,2}-[A-Z]{1,2}$', r'^[A-Z]{1,2}\d{0,2}(\.\d{0,2})?$',
                 r'^[A-Z]{1,2}-\d+(-\d+)?$', r'^[A-Za-z]\s\d{1,2}$', r'^[A-Za-z]\s[A-Za-z]$', r'^[A-Za-z]{1,2}\.\d+$'],
        shape_coef_min=1/1.5,
        shape_coef_max=1.5,
        iou_thr=0.5),
    filter_by_color_nin=dict(colors=[None, '#FFFFFF']),
    find_grid_lines_of_object=dict(tol=4,
                                   top_n_to_cluster=100),
    filter_by_grids_areas=dict(iou_thr=0.9),
    filter_by_line_number=dict(max_conn=20),
    model_filtering=dict(threshold=0.5,
                         thr_ratio=0.33,
                         grids_probable_perc=0.5,
                         model_path="/app/model/lgb.model"),
    filter_key_plan_grids=dict(area_ratio=0.9,
                               max_dist_to_text=0.3,
                               max_dist_to_line=1),
    find_closest_grid_to_center=dict(extend_factor=400),
    limit_grid_line_by_color=dict(tol=1),
    limit_grid_line_by_dots=dict(n_std=13))

JUNCTION_BOX_DETECTION_CONF = dict(filter_by_color_in=dict(colors=['#000000']),
                                   get_closed_objects=dict(tol=2),
                                   filter_objects_by_proportion=dict(thr=0.3, eps=1e-6),
                                   get_text_inside_objects=dict(iou_thr=0.8),
                                   classify_by_exact_text_inside_object=dict(
                                       heuristic={'junction_box': 'J',
                                                  'fsd': 'FSD',
                                                  'vfd': 'VFD',
                                                  'combination fire alarm speaker visual': 'FV',
                                                  'fire alarm horn': 'H',
                                                  'magnetic lock': 'ML',
                                                  'fire alarm audio unit': 'F',
                                                  'door contact': 'DC',
                                                  'television outlet': 'TV',
                                                  'door release': 'DR',
                                                  'door holder': 'DH',
                                                  'smoke detector': 'S',
                                                  'motor': 'M',
                                                  'cctv camera': 'CCTV',
                                                  'electrical connection': 'E'},
                                       overall_class='junction box'
                                   ),
                                   classify_by_exact_text_only=dict(heuristic={'fsd': 'FSD'}),
                                   detect_switches=True,
                                   filter_disconnected_switches=dict(max_area=8000,
                                                                     max_bbox_ls_trigger_l_ratio=0.8,
                                                                     min_bbox_ls_trigger_l_ratio=0.1,
                                                                     min_max_trigger_ls_ratio=0.1)
                                   )

SLD_CLOSED_OBJECTS_CONF = dict(filter_by_color_nin=dict(colors=[None, '#FFFFFF']),
                               del_by_existence_lines=dict(threshold=0.1),
                               split_all_lines_by_intersections=dict(for_sld=True,
                                                                     tol1=1,
                                                                     tol2=2),
                               get_closed_objects=dict(tol=5),
                               del_by_existence_objects=dict(threshold_line=0.5,
                                                             threshold_obj=0.5))
SLD_OBJECTS_MERGING_CONF = dict(pairwise_delete_objects_by_area=dict(iou_thr=0.2))
