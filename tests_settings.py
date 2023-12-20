# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import os
import string
import logging
from pathlib import Path
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from dotenv import load_dotenv
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
from src_logging import log_config
# >>>> ********************************************************************************

# ________________________________________________________________________________
# --- FASTAPI TEST SETTINGS ---
PORT = 8000

if os.getenv("ENVIRONMENT") == "DEVELOPMENT":
    PORT = os.getenv("DEV_APP_PORT")

if os.getenv("ENVIRONMENT") == "STAGE":
    PORT = os.getenv("STAGE_APP_PORT")

if os.getenv("ENVIRONMENT") == "PRODUCTION":
    PORT = os.getenv("PROD_APP_PORT")

PREFIX = "/v1"
BASE_URL = f"http://localhost:{PORT}{PREFIX}"


# ________________________________________________________________________________
# --- PAYLOAD ---

TEST_DIRS = dict(
    closed_objects_sld=dict(
        pdf_dir="tmp/integration_test/",
        json_dir="tmp/sld/2/",
        out_dir="tmp/sld/2/"),
    detect_jb=dict(
        json_dir="tmp/835539555135300/",
        out_dir="tmp/835539555135300/"),
    grid_lines_detection=dict(
        json_dir="tmp/836261982596700/",
        out_dir="tmp/836261982596700/"),
    nms_objects_refinement=dict(
        json_dir="tmp/835539555135300/",
        out_dir="tmp/835539555135300/"),
    closed_objects_merging_sld=dict(
        json_dir="tmp/179917879821900/",
        out_dir="tmp/sld/4/")
)

BUCKET_NAME = "drawer-dev-project-files"

TEST_CLOSED_OBJECTS_SLD_S3 = dict(
    link='/closed-objects-sld-s3/',
    json_data=dict(
        files=dict(
            pdf_part=dict(
                pdf_file=dict(
                    file_key=TEST_DIRS["closed_objects_sld"]["pdf_dir"]+"SLD 5 - tag.pdf")
            ),
            json_part=dict(
                lines=dict(
                    file_key=TEST_DIRS["closed_objects_sld"]["json_dir"]+"SLD_5_lines.json"),
                cubic_lines=dict(
                    file_key=TEST_DIRS["closed_objects_sld"]["json_dir"]+"SLD_5_cubic_lines.json"),
                circles=dict(
                    file_key=TEST_DIRS["closed_objects_sld"]["json_dir"]+"SLD_5_circles.json"),
                quads=dict(
                    file_key=TEST_DIRS["closed_objects_sld"]["json_dir"]+"SLD_5_quads.json"),
                rectangles=dict(
                    file_key=TEST_DIRS["closed_objects_sld"]["json_dir"]+"SLD_5_rectangles.json"),
            )
        ),
        out_files=dict(
            json_part=dict(
                closed_objects=dict(
                    file_key=TEST_DIRS["closed_objects_sld"]["out_dir"]+"SLD_5_closed_objects.json"),
                lines=dict(
                    file_key=TEST_DIRS["closed_objects_sld"]["out_dir"]+"SLD_5_closed_lines.json")
            )
        ),
        s3_bucket_name=BUCKET_NAME,
        page_num=0
    )
)

TEST_CLOSED_OBJECTS_SLD_JSON = dict(
    link='/closed-objects-sld-json/',
    s3_bucket_name=BUCKET_NAME,
    file_keys=dict(
        pdf_file=TEST_CLOSED_OBJECTS_SLD_S3["json_data"]["files"]["pdf_part"]["pdf_file"]["file_key"],
        lines=TEST_CLOSED_OBJECTS_SLD_S3["json_data"]["files"]["json_part"]["lines"]["file_key"],
        cubic_lines=TEST_CLOSED_OBJECTS_SLD_S3["json_data"]["files"]["json_part"]["cubic_lines"]["file_key"],
        circles=TEST_CLOSED_OBJECTS_SLD_S3["json_data"]["files"]["json_part"]["circles"]["file_key"],
        quads=TEST_CLOSED_OBJECTS_SLD_S3["json_data"]["files"]["json_part"]["quads"]["file_key"],
        rectangles=TEST_CLOSED_OBJECTS_SLD_S3["json_data"]["files"]["json_part"]["rectangles"]["file_key"]
    ),
    json_data=dict(
        page_num=0)
)

TEST_DETECT_JB_S3 = dict(
    link='/detect-jb-json-s3/',
    json_data=dict(
        files=dict(
            lines=dict(
                file_key=TEST_DIRS["detect_jb"]["json_dir"]+"small-sample.pdflinesExtraction.json"),
            cubic_lines=dict(
                file_key=TEST_DIRS["detect_jb"]["json_dir"]+"small-sample.pdfcubicLinesExtraction.json"),
            circles=dict(
                file_key=TEST_DIRS["detect_jb"]["json_dir"]+"small-sample.pdfcirclesExtraction.json"),
            rectangles=dict(
                file_key=TEST_DIRS["detect_jb"]["json_dir"]+"small-sample.pdfrectanglesExtraction.json"),
            quads=dict(
                file_key=TEST_DIRS["detect_jb"]["json_dir"]+"small-sample.pdfquadsExtraction.json"),
            parsed_text=dict(
                file_key=TEST_DIRS["detect_jb"]["json_dir"]+"small-sample.pdfdrTextExtractionS3Step.json")
        ),
        s3_bucket_name=BUCKET_NAME,
        out_s3_file_key=TEST_DIRS["detect_jb"]["out_dir"]+"small-sample.pdfdetectJunctionBoxesS3Step_for_ui.json"
    )
)

TEST_DETECT_JB_JSON = dict(
    link='/detect-jb-json/',
    s3_bucket_name=BUCKET_NAME,
    file_keys=dict(
        lines=TEST_DETECT_JB_S3["json_data"]["files"]["lines"]["file_key"],
        cubic_lines=TEST_DETECT_JB_S3["json_data"]["files"]["cubic_lines"]["file_key"],
        circles=TEST_DETECT_JB_S3["json_data"]["files"]["circles"]["file_key"],
        rectangles=TEST_DETECT_JB_S3["json_data"]["files"]["rectangles"]["file_key"],
        quads=TEST_DETECT_JB_S3["json_data"]["files"]["quads"]["file_key"],
        parsed_text=TEST_DETECT_JB_S3["json_data"]["files"]["parsed_text"]["file_key"]
    )
)

TEST_GRID_LINES_DETECTION_S3 = dict(
    link='/grid-lines-detection-json-s3/',
    json_data=dict(
        files=dict(
            lines=dict(
                file_key=TEST_DIRS["grid_lines_detection"]["json_dir"]+"Project 2 - Overall.pdflinesExtraction.json"),
            cubic_lines=dict(
                file_key=TEST_DIRS["grid_lines_detection"]["json_dir"]+"Project 2 - Overall.pdfcubicLinesExtraction.json"),
            circles=dict(
                file_key=TEST_DIRS["grid_lines_detection"]["json_dir"]+"Project 2 - Overall.pdfcirclesExtraction.json"),
            rectangles=dict(
                file_key=TEST_DIRS["grid_lines_detection"]["json_dir"]+"Project 2 - Overall.pdfrectanglesExtraction.json"),
            quads=dict(
                file_key=TEST_DIRS["grid_lines_detection"]["json_dir"]+"Project 2 - Overall.pdfquadsExtraction.json"),
            parsed_text=dict(
                file_key=TEST_DIRS["grid_lines_detection"]["json_dir"]+"Project 2 - Overall.pdfstitchingS3TextExtractionStep.json")
        ),
        s3_bucket_name=BUCKET_NAME,
        out_s3_file_key=TEST_DIRS["grid_lines_detection"]["out_dir"]+"Project 2 - Overall.pdfstitchingS3GridLinesExtractionStep.json"
    )
)

TEST_GRID_LINES_DETECTION_JSON = dict(
    link='/grid-lines-detection-json/',
    s3_bucket_name=BUCKET_NAME,
    file_keys=dict(
        lines=TEST_GRID_LINES_DETECTION_S3["json_data"]["files"]["lines"]["file_key"],
        cubic_lines=TEST_GRID_LINES_DETECTION_S3["json_data"]["files"]["cubic_lines"]["file_key"],
        circles=TEST_GRID_LINES_DETECTION_S3["json_data"]["files"]["circles"]["file_key"],
        rectangles=TEST_GRID_LINES_DETECTION_S3["json_data"]["files"]["rectangles"]["file_key"],
        quads=TEST_GRID_LINES_DETECTION_S3["json_data"]["files"]["quads"]["file_key"],
        parsed_text=TEST_GRID_LINES_DETECTION_S3["json_data"]["files"]["parsed_text"]["file_key"]
    )
)

TEST_NMS_OBJECTS_REFINEMENT_S3 = dict(
    link='/nms-objects-refinement-json-s3/',
    json_data=dict(
        files=dict(
            first_detections=dict(
                file_key=TEST_DIRS["nms_objects_refinement"]["json_dir"]+"small-sample.pdfdetectReceptaclesS3Step.json"),
            second_detections=dict(
                file_key=TEST_DIRS["nms_objects_refinement"]["json_dir"]+"small-sample.pdfdetectJunctionBoxesS3Step.json")
        ),
        s3_bucket_name=BUCKET_NAME,
        out_s3_file_key=TEST_DIRS["nms_objects_refinement"]["out_dir"]+"small-sample.pdf_result_.json"
    )
)

TEST_NMS_OBJECTS_REFINEMENT_JSON = dict(
    link='/nms-objects-refinement-json/',
    s3_bucket_name=BUCKET_NAME,
    file_keys=dict(
        first_detections=TEST_NMS_OBJECTS_REFINEMENT_S3["json_data"]["files"]["first_detections"]["file_key"],
        second_detections=TEST_NMS_OBJECTS_REFINEMENT_S3["json_data"]["files"]["second_detections"]["file_key"]
    )
)

TEST_CLOSED_OBJECTS_MERGING_SLD_S3 = dict(
    link='/closed-objects-merging-sld-s3/',
    json_data=dict(
        files=dict(
            json_part=dict(
                filtered_lines=dict(
                    file_key=TEST_DIRS["closed_objects_merging_sld"]["json_dir"]+"SLD 5 - tag.pdf_lines_detectObjectsSLDDetectClosedObjectsS3Step.json"),
                closed_objects=dict(
                    file_key=TEST_DIRS["closed_objects_merging_sld"]["json_dir"]+"SLD 5 - tag.pdf_closed_objects_detectObjectsSLDDetectClosedObjectsS3Step.json"),
                detected_objects=dict(
                    file_key=TEST_DIRS["closed_objects_merging_sld"]["json_dir"]+"SLD 5 - tag.pdfdetectObjectsSLDDetectByModelS3Step.json"),
                junction_boxes=dict(
                    file_key=TEST_DIRS["closed_objects_merging_sld"]["json_dir"]+"SLD 5 - tag.pdfdetectObjectsSLDJunctionBoxesAndDisconnectedSwitchesS3Step.json")
            )
        ),
        out_files=dict(
            json_part=dict(
                merged_objects=dict(
                    file_key=TEST_DIRS["closed_objects_merging_sld"]["out_dir"]+"SLD_5_out_s3_file_key_merged_objects.json")
            )
        ),
        s3_bucket_name=BUCKET_NAME
    )
)

TEST_CLOSED_OBJECTS_MERGING_SLD_JSON = dict(
    link='/closed-objects-merging-sld-json/',
    s3_bucket_name=BUCKET_NAME,
    file_keys=dict(
        filtered_lines=TEST_CLOSED_OBJECTS_MERGING_SLD_S3["json_data"]["files"]["json_part"]["filtered_lines"]["file_key"],
        closed_objects=TEST_CLOSED_OBJECTS_MERGING_SLD_S3["json_data"]["files"]["json_part"]["closed_objects"]["file_key"],
        detected_objects=TEST_CLOSED_OBJECTS_MERGING_SLD_S3["json_data"]["files"]["json_part"]["detected_objects"]["file_key"],
        junction_boxes=TEST_CLOSED_OBJECTS_MERGING_SLD_S3["json_data"]["files"]["json_part"]["junction_boxes"]["file_key"]
    )
)
