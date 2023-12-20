# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
import os
import io
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import FastAPI
from fastapi.testclient import TestClient
# import httpx
import pytest
import ujson as json
# from pydantic import BaseModel, Field
from typing import List
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
from main import app
from src_logging import log_config
import tests_settings
from src_tests.unit_tests import assertion_s3
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.DEBUG)

client = TestClient(app)


# ________________________________________________________________________________
# >>>> </> TEST - CLOSED OBJECTS ENDPOINT - S3 </>
def test_closed_objects_endpoint_s3():
    test_data = tests_settings.TEST_CLOSED_OBJECTS_SLD_S3
    assertion_s3(test_data, 200)


# ________________________________________________________________________________
# >>>> </> TEST - DETECT JB ENDPOINT JSON - S3 </>
def test_ds_jb_endpoint_json_s3():
    test_data = tests_settings.TEST_DETECT_JB_S3
    assertion_s3(test_data, 200)


# ________________________________________________________________________________
# >>>> </> TEST - FIND GRID LINES ENDPOINT JSON - S3 </>
def test_grid_lines_endpoint_json_s3():
    test_data = tests_settings.TEST_GRID_LINES_DETECTION_S3
    assertion_s3(test_data, 200)


# ________________________________________________________________________________
# >>>> </> TEST - NMS OBJECTS REFINEMENT ENDPOINT JSON - S3 </>
def test_nms_objects_refinement_endpoint_s3():
    test_data = tests_settings.TEST_NMS_OBJECTS_REFINEMENT_S3
    assertion_s3(test_data, 200)


# ________________________________________________________________________________
# >>>> </> TEST - CLOSED OBJECTS MERGING SLD ENDPOINT - S3 </>
def test_closed_objects_merging_endpoint_s3():
    test_data = tests_settings.TEST_CLOSED_OBJECTS_MERGING_SLD_S3
    assertion_s3(test_data, 200)


# ________________________________________________________________________________
# >>>> </> TEST - HEALTHCHECK </>
def test_healthcheck_get():
    response = client.get("/healthcheck/")
    assert response.status_code == 200
    assert response.json() == {
        "healthcheck": "API Status 200"
    }



