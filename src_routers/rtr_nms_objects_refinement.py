# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
import io
from copy import deepcopy
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import APIRouter, HTTPException, Form
from fastapi import status, UploadFile, File, Response
from fastapi.responses import UJSONResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import ujson as json
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
# ---- CONFIG ----
# import settings
from src_logging import log_config
# ---- UTILS ----
from src_utils.loading_utils import load_pdf
from src_utils.zipping_utils import FileProc
from src_utils.plotting_uitls import plot_detection_result
from src_utils.aws_utils import S3FileOps
# ---- PROCESSES ----
from src_processes.proc_nms_multi_model import merge_and_NMS
# ---- REQUEST MODELS ----
from src_routers.request_models.pydantic_models_s3 import NmsObjectRefinementFilesDataS3
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.INFO)

# ________________________________________________________________________________
# --- FastAPI ROUTER ---
nms_objects_refinement_rtr = APIRouter(prefix="/v1")


class NMSData(BaseModel):
    first_detections: dict
    second_detections: dict


# ________________________________________________________________________________
@nms_objects_refinement_rtr.post(path="/nms-objects-refinement-zip/",
                                 responses={200: {}, 500: {}, 503: {}},
                                 status_code=status.HTTP_200_OK,
                                 response_class=StreamingResponse,
                                 tags=["NMS Objects Refinement"],
                                 summary="Apply NMS for objects and return results in zip")
async def nms_objects_refinement_endpoint_zip(first_file_detections: UploadFile = File(...),
                                              second_file_detections: UploadFile = File(...),
                                              pdf_file: UploadFile = File(...),
                                              page_num: int = Form(0),
                                              visualize_results: bool = Form(True)) -> StreamingResponse:
    if not first_file_detections.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {first_file_detections.filename} does not end with .json')

    if not second_file_detections.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {second_file_detections.filename} does not end with .json')
    # load jsons
    objects_first_file = json.loads(await first_file_detections.read())
    objects_second_file = json.loads(await second_file_detections.read())
    # gather togather
    result_detection_obj = deepcopy(objects_first_file)
    result_detection_obj['data'] = result_detection_obj['data'] + objects_second_file['data']
    # get pdf file
    _, __, img_array, ___ = load_pdf(pdf_file, page_num)
    logger.info('Started merging and NMS')
    res_dict = merge_and_NMS(result_detection_obj)
    logger.info('Finished')

    if visualize_results:
        img_array = plot_detection_result(img_array, res_dict)
        json_content = {"result_detection": res_dict}
        img_content = {"test_img": img_array}
        resp_content, resp_media_type = FileProc.get_zip_with_json_and_img(json_content=json_content,
                                                                           img_content=img_content)

    else:
        json_content = {"result_detection": res_dict}
        resp_content, resp_media_type = FileProc.get_zip_with_json(json_content=json_content)

    return StreamingResponse(content=resp_content,
                             status_code=status.HTTP_200_OK,
                             media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=nms_result.zip"})


# ________________________________________________________________________________
@nms_objects_refinement_rtr.post(path="/nms-objects-refinement-json/",
                                 responses={200: {}, 500: {}, 503: {}},
                                 status_code=status.HTTP_200_OK,
                                 response_class=StreamingResponse,
                                 tags=["NMS Objects Refinement"],
                                 summary="Apply NMS for objects and return results in json")
async def nms_objects_refinement_endpoint_json(item: NMSData) -> UJSONResponse:
    first_detections = item.first_detections
    second_detections = item.second_detections

    result_detection_obj = deepcopy(first_detections)
    result_detection_obj['data'] = result_detection_obj['data'] + second_detections['data']
    # actual process
    logger.info('Started merging and NMS')
    res_dict = merge_and_NMS(result_detection_obj)
    logger.info('Finished')

    return UJSONResponse(content=res_dict)


# ________________________________________________________________________________
@nms_objects_refinement_rtr.post(path="/nms-objects-refinement-json-s3/",
                                 responses={200: {}, 500: {}, 503: {}},
                                 status_code=status.HTTP_200_OK,
                                 response_class=StreamingResponse,
                                 tags=["NMS Objects Refinement", "S3"],
                                 summary="Apply NMS for objects and return results in json")
async def nms_objects_refinement_endpoint_s3(files_data: NmsObjectRefinementFilesDataS3) -> Response:
    # ________________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    s3 = S3FileOps(s3_bucket_name=files_data.s3_bucket_name)

    # --- ITERATE OVER JSON-FILES KEYS (attributes of | GridLinesDetectionFiles | Pydantic model.
    # --- "json_data" is a Dict of Dicts with multiple JSONs data payloads that
    # --- are accessible by keys equivalent to fields in respective Pydantic model ---
    json_data = {}
    for attr_name, json_file_attrs in files_data.files.dict().items():
        json_data[attr_name] = s3.get_json_file_data(s3_file_key=json_file_attrs["file_key"])

    first_detections = json_data["first_detections"]
    second_detections = json_data["second_detections"]

    # ________________________________________________________________________________
    # --- GATHER TOGETHER ---
    result_detection_obj = deepcopy(first_detections)
    result_detection_obj['data'] = result_detection_obj['data'] + second_detections['data']

    # --- ACTUAL PROCESS ---
    logger.info('Started merging and NMS')
    result_dict = merge_and_NMS(result_detection_obj)
    logger.info('Finished')

    # ________________________________________________________________________________
    # --- CONVERT JSON TO BYTES STREAM ---
    json_payload = json.dumps(result_dict)
    json_byte_stream = io.BytesIO(json_payload.encode("utf-8"))

    # --- UPLOAD FILE TO AWS S3 BUCKET ---
    try:
        s3_upload_status = s3.upload_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                              s3_file_key=files_data.out_s3_file_key,
                                              file_byte_stream=json_byte_stream)
        if not s3_upload_status:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"ERROR -> S3 upload status: {s3_upload_status}")
        return Response(status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"ERROR -> Failed to upload file to S3. Error: {e}")
