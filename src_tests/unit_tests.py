# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
# TEMPORARY
import os
import io
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
# from fastapi import FastAPI
from fastapi.testclient import TestClient
# import httpx
import pytest
import ujson as json
# from pydantic import BaseModel, Field
# from typing import List
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
from main import app
from src_logging import log_config
from src_utils.aws_utils import S3FileOps
import tests_settings
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.DEBUG)

client = TestClient(app)

dev_username = "drawer"
dev_password = "Y4AuMasf"
base_url = tests_settings.BASE_URL


# ________________________________________________________________________________
# >>>> </> TEST - CLOSED OBJECTS ENDPOINT - S3 - FILE NOT FOUND </>
def test_closed_objects_endpoint_s3_file_not_found():
    test_data = tests_settings.TEST_CLOSED_OBJECTS_SLD_S3
    test_data["json_data"]["files"]["pdf_part"]["pdf_file"]["file_key"] = "wrong_file.pdf"
    assertion_s3(test_data, 503)


# ________________________________________________________________________________
# >>>> </> TEST - DETECT JB ENDPOINT JSON - S3 - FILE NOT FOUND </>
def test_ds_jb_endpoint_json_s3_file_not_found():
    test_data = tests_settings.TEST_DETECT_JB_S3
    test_data["json_data"]["files"]["lines"]["file_key"] = "wrong_file.json"
    assertion_s3(test_data, 503)


# ________________________________________________________________________________
# >>>> </> TEST - FIND GRID LINES ENDPOINT JSON - S3 - FILE NOT FOUND </>
def test_grid_lines_endpoint_json_s3_file_not_found():
    test_data = tests_settings.TEST_GRID_LINES_DETECTION_S3
    test_data["json_data"]["files"]["lines"]["file_key"] = "wrong_file.json"
    assertion_s3(test_data, 503)


# ________________________________________________________________________________
# >>>> </> TEST - NMS OBJECTS REFINEMENT ENDPOINT JSON - S3 - FILE NOT FOUND </>
def test_nms_objects_refinement_endpoint_s3_file_not_found():
    test_data = tests_settings.TEST_NMS_OBJECTS_REFINEMENT_S3
    test_data["json_data"]["files"]["first_detections"]["file_key"] = "wrong_file.json"
    assertion_s3(test_data, 503)


# ________________________________________________________________________________
# >>>> </> TEST - CLOSED OBJECTS MERGING SLD ENDPOINT - S3 - FILE NOT FOUND </>
def test_closed_objects_merging_endpoint_s3_file_not_found():
    test_data = tests_settings.TEST_CLOSED_OBJECTS_MERGING_SLD_S3
    test_data["json_data"]["files"]["json_part"]["filtered_lines"]["file_key"] = "wrong_file.json"
    assertion_s3(test_data, 503)


# ________________________________________________________________________________
# >>>> </> TEST - CLOSED OBJECTS ENDPOINT JSON - JSON </>
def test_closed_objects_endpoint_json():
    test_data = tests_settings.TEST_CLOSED_OBJECTS_SLD_JSON
    link = base_url + test_data["link"]

    pdf_data = download_json_s3(test_data["file_keys"]["pdf_file"],
                                test_data["s3_bucket_name"])
    lines_data = download_json_s3(test_data["file_keys"]["lines"],
                                  test_data["s3_bucket_name"])
    cubic_lines_data = download_json_s3(test_data["file_keys"]["cubic_lines"],
                                        test_data["s3_bucket_name"])
    circles_data = download_json_s3(test_data["file_keys"]["circles"],
                                    test_data["s3_bucket_name"])
    rectangles_data = download_json_s3(test_data["file_keys"]["rectangles"],
                                       test_data["s3_bucket_name"])
    quads_data = download_json_s3(test_data["file_keys"]["quads"],
                                  test_data["s3_bucket_name"])

    files = {
        "pdf_file": (pdf_data.name, pdf_data, "application/pdf"),
        "lines_file": (lines_data.name, lines_data, "application/json"),
        "cubic_lines_file": (cubic_lines_data.name, cubic_lines_data, "application/json"),
        "circles_file": (circles_data.name, circles_data, "application/json"),
        "rectangles_file": (rectangles_data.name, rectangles_data, "application/json"),
        "quads_file": (quads_data.name, quads_data, "application/json")
    }

    json_data = test_data["json_data"]

    response = client.post(url=link,
                           files=files,
                           data=json_data)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE + CONTENT TYPE ---
    assertion_json(response,
                   status_code=200)


# ________________________________________________________________________________
# >>>> </> TEST - DETECT JB ENDPOINT JSON - JSON </>
def test_ds_jb_endpoint_json():
    test_data = tests_settings.TEST_DETECT_JB_JSON
    link = base_url + test_data["link"]

    lines_data = download_json_s3(test_data["file_keys"]["lines"],
                                  test_data["s3_bucket_name"])
    cubic_lines_data = download_json_s3(test_data["file_keys"]["cubic_lines"],
                                        test_data["s3_bucket_name"])
    circles_data = download_json_s3(test_data["file_keys"]["circles"],
                                    test_data["s3_bucket_name"])
    rectangles_data = download_json_s3(test_data["file_keys"]["rectangles"],
                                       test_data["s3_bucket_name"])
    quads_data = download_json_s3(test_data["file_keys"]["quads"],
                                  test_data["s3_bucket_name"])
    parsed_text_data = download_json_s3(test_data["file_keys"]["parsed_text"],
                                        test_data["s3_bucket_name"])

    json_data = {
        "lines": lines_data,
        "cubic_lines": cubic_lines_data,
        "circles": circles_data,
        "rectangles": rectangles_data,
        "quads": quads_data,
        "parsed_text": parsed_text_data
    }
    json_payload = json.dumps(json_data)

    response = client.post(url=link,
                           content=json_payload)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE + CONTENT TYPE ---
    assertion_json(response,
                   status_code=200)


# ________________________________________________________________________________
# >>>> </> TEST - FIND GRID LINES ENDPOINT JSON - JSON </>
def test_grid_lines_endpoint_json():
    test_data = tests_settings.TEST_GRID_LINES_DETECTION_JSON
    link = base_url + test_data["link"]

    lines_data = download_json_s3(test_data["file_keys"]["lines"],
                                  test_data["s3_bucket_name"])
    cubic_lines_data = download_json_s3(test_data["file_keys"]["cubic_lines"],
                                        test_data["s3_bucket_name"])
    circles_data = download_json_s3(test_data["file_keys"]["circles"],
                                    test_data["s3_bucket_name"])
    rectangles_data = download_json_s3(test_data["file_keys"]["rectangles"],
                                       test_data["s3_bucket_name"])
    quads_data = download_json_s3(test_data["file_keys"]["quads"],
                                  test_data["s3_bucket_name"])
    parsed_text_data = download_json_s3(test_data["file_keys"]["parsed_text"],
                                        test_data["s3_bucket_name"])

    json_data = {
        "lines": lines_data,
        "cubic_lines": cubic_lines_data,
        "circles": circles_data,
        "rectangles": rectangles_data,
        "quads": quads_data,
        "parsed_text": parsed_text_data
    }
    json_payload = json.dumps(json_data)

    response = client.post(url=link,
                           content=json_payload)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE + CONTENT TYPE ---
    assertion_json(response,
                   status_code=200)


# ________________________________________________________________________________
# >>>> </> TEST - NMS OBJECTS REFINEMENT ENDPOINT JSON - JSON </>
def test_nms_objects_refinement_endpoint_json():
    test_data = tests_settings.TEST_NMS_OBJECTS_REFINEMENT_JSON
    link = base_url + test_data["link"]

    first_detections_data = download_json_s3(test_data["file_keys"]["first_detections"],
                                             test_data["s3_bucket_name"])
    second_detections_data = download_json_s3(test_data["file_keys"]["second_detections"],
                                              test_data["s3_bucket_name"])

    json_data = {
        "first_detections": first_detections_data,
        "second_detections": second_detections_data
    }
    json_payload = json.dumps(json_data)

    response = client.post(url=link,
                           content=json_payload)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE + CONTENT TYPE ---
    assertion_json(response,
                   status_code=200)


# ________________________________________________________________________________
# >>>> </> TEST - CLOSED OBJECTS MERGING SLD ENDPOINT JSON - JSON </>
def test_closed_objects_merging_endpoint_json():
    test_data = tests_settings.TEST_CLOSED_OBJECTS_MERGING_SLD_JSON
    link = base_url + test_data["link"]

    filtered_lines_data = download_file_s3(test_data["file_keys"]["filtered_lines"],
                                           test_data["s3_bucket_name"])
    closed_objects_data = download_file_s3(test_data["file_keys"]["closed_objects"],
                                           test_data["s3_bucket_name"])
    detected_objects_data = download_file_s3(test_data["file_keys"]["detected_objects"],
                                             test_data["s3_bucket_name"])
    junction_boxes_data = download_file_s3(test_data["file_keys"]["junction_boxes"],
                                           test_data["s3_bucket_name"])

    files = {
        "filtered_lines": (filtered_lines_data.name, filtered_lines_data, "application/json"),
        "closed_objects": (closed_objects_data.name, closed_objects_data, "application/json"),
        "detected_objects": (detected_objects_data.name, detected_objects_data, "application/json"),
        "junction_boxes": (junction_boxes_data.name, junction_boxes_data, "application/json")
    }

    response = client.post(url=link,
                           files=files)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE + CONTENT TYPE ---
    assertion_json(response,
                   status_code=200)


def assertion_json(response,
                   status_code: int):
    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE ---
    assert response.status_code == status_code
    logger.info(f"- RESPONSE STATUS CODE - {response.status_code}")

    # ________________________________________________________________________________
    # --- CHECK RESPONSE CONTENT TYPE ---
    assert response.headers["content-type"] == "application/json"
    content_type = response.headers["content-type"]
    logger.info(f"- RESPONSE CONTENT TYPE - {content_type}")


# ________________________________________________________________________________
# >>>> </> TEST ASSERTION - S3 </>
def assertion_s3(test_data,
                 status_code):
    link = base_url + test_data["link"]

    json_data = test_data["json_data"]
    json_payload = json.dumps(json_data)

    response = client.post(url=link,
                           content=json_payload)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE ---
    assert response.status_code == status_code
    logger.info(f"- RESPONSE STATUS CODE - {response.status_code}")


def download_n_save_file_s3(file_key, s3_bucket_name):
    # _____________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    logger.info("- CONNECTING TO AWS S3 BUCKET -")
    s3 = S3FileOps(s3_bucket_name=s3_bucket_name)

    # --- DOWNLOAD FILE FROM S3 ---
    file_data = s3.download_file_obj(s3_bucket_name, file_key)

    # --- SAVING FILE ---
    logger.info("- 2 - SAVING FILE -")
    filename = file_key.split('/')[-1]

    with open(filename, "wb") as f:
        f.write(file_data.getvalue())


def download_file_s3(file_key, s3_bucket_name):
    # _____________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    logger.info("- CONNECTING TO AWS S3 BUCKET -")
    s3 = S3FileOps(s3_bucket_name=s3_bucket_name)

    # --- DOWNLOAD FILE FROM S3 ---
    file_data = s3.download_file_obj(s3_bucket_name, file_key)
    file_data.name = file_key.split('/')[-1]

    return file_data


def download_json_s3(file_key, s3_bucket_name):
    # _____________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    logger.info("- CONNECTING TO AWS S3 BUCKET -")
    s3 = S3FileOps(s3_bucket_name=s3_bucket_name)

    # --- DOWNLOAD FILE FROM S3 ---
    file_data = s3.get_json_file_data(file_key)

    return file_data
