# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import APIRouter, HTTPException, Form
from fastapi import status, UploadFile, File, Response
from fastapi.responses import UJSONResponse
from starlette.responses import StreamingResponse
import ujson as json
from typing import Optional
import settings
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
# ---- CONFIG ----
from src_logging import log_config
# ---- UTILS ----
from src_utils.loading_utils import load_pdf
from src_utils.zipping_utils import FileProc
from src_utils.plotting_uitls import plot_closed_objects_sld
from src_utils.aws_utils import S3FileOps
from dataclasses import dataclass
# ---- PROCESSES ----
from src_processes.merge_objects_sld import merge_objects_sld
# ---- REQUEST MODELS ----
from src_routers.request_models.mdl_closed_objects_merging_sld import ClosedObjectsMergingSldS3FilesData
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.INFO)

# ________________________________________________________________________________
# --- FastAPI ROUTER ---
closed_objects_merging_sld_rtr = APIRouter(prefix="/v1")


# ________________________________________________________________________________
@closed_objects_merging_sld_rtr.post(path="/closed-objects-merging-sld-zip/",
                                     responses={200: {}, 500: {}, 503: {}},
                                     status_code=status.HTTP_200_OK,
                                     response_class=StreamingResponse,
                                     tags=["Closed objects merging"],
                                     summary="Closed objects merging for SLD in zip")
async def closed_objects_merging_endpoint_zip(filtered_lines_file: UploadFile = File(...),
                                              closed_objects_file: UploadFile = File(...),
                                              detected_objects_file: UploadFile = File(...),
                                              junction_boxes_file: UploadFile = File(...),
                                              pdf_file: UploadFile = File(...),
                                              page_num: int = Form(0),
                                              visualize_results: bool = Form(True)) -> StreamingResponse:
    if not filtered_lines_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {filtered_lines_file.filename} does not end with .json')

    if not closed_objects_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {closed_objects_file.filename} does not end with .json')

    if not detected_objects_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {detected_objects_file.filename} does not end with .json')

    if not junction_boxes_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {junction_boxes_file.filename} does not end with .json')

    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(404, f'File with filename {pdf_file.filename} does not end with .pdf')


    # load jsons
    filtered_lines = json.loads(await filtered_lines_file.read())
    closed_objects = json.loads(await closed_objects_file.read())
    detected_objects = json.loads(await detected_objects_file.read())
    junction_boxes = json.loads(await junction_boxes_file.read())

    # get pdf file
    _, __, img_array, ___ = load_pdf(pdf_file, page_num)
    logger.info('Started merging of closed objects for SLD')
    merged_objects = merge_objects_sld(filtered_lines=filtered_lines,
                                       closed_objects=closed_objects,
                                       detected_objects=detected_objects,
                                       junction_boxes=junction_boxes,
                                       config=settings.SLD_OBJECTS_MERGING_CONF)
    logger.info('Finished')

    if visualize_results:
        img_array = plot_closed_objects_sld(filtered_lines, merged_objects)
        json_content = {"merged_objects": merged_objects}
        img_content = {"test_img": img_array}
        resp_content, resp_media_type = FileProc.get_zip_with_json_and_img(json_content=json_content,
                                                                           img_content=img_content)

    else:
        json_content = {"merged_objects": merged_objects}
        resp_content, resp_media_type = FileProc.get_zip_with_json(json_content=json_content)

    return StreamingResponse(content=resp_content,
                             status_code=status.HTTP_200_OK,
                             media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=merged_objects.zip"})


# ________________________________________________________________________________
@closed_objects_merging_sld_rtr.post(path="/closed-objects-merging-sld-json/",
                                     responses={200: {}, 500: {}, 503: {}},
                                     status_code=status.HTTP_200_OK,
                                     response_class=UJSONResponse,
                                     tags=["Closed objects merging"],
                                     summary="Closed objects merging for SLD in json")
async def closed_objects_merging_endpoint_json(filtered_lines_file: UploadFile = File(...),
                                               closed_objects_file: UploadFile = File(...),
                                               detected_objects_file: UploadFile = File(...),
                                               junction_boxes_file: UploadFile = File(...)
                                               ) -> UJSONResponse:
    if not filtered_lines_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {filtered_lines_file.filename} does not end with .json')

    if not closed_objects_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {closed_objects_file.filename} does not end with .json')

    if not detected_objects_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {detected_objects_file.filename} does not end with .json')

    if not junction_boxes_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {junction_boxes_file.filename} does not end with .json')

    # load jsons
    filtered_lines = json.loads(await filtered_lines_file.read())
    closed_objects = json.loads(await closed_objects_file.read())
    detected_objects = json.loads(await detected_objects_file.read())
    junction_boxes = json.loads(await junction_boxes_file.read())

    # get pdf file
    logger.info('Started merging of closed objects for SLD')
    merged_objects = merge_objects_sld(filtered_lines=filtered_lines,
                                       closed_objects=closed_objects,
                                       detected_objects=detected_objects,
                                       junction_boxes=junction_boxes,
                                       config=settings.SLD_OBJECTS_MERGING_CONF)
    logger.info('Finished')

    return UJSONResponse(content=merged_objects)


@dataclass(slots=True)
class ClosedObjectsMergingSldS3JsonData:
    detected_objects:   dict = None
    junction_boxes:     dict = None
    filtered_lines:     dict = None
    closed_objects:     dict = None


# ________________________________________________________________________________
@closed_objects_merging_sld_rtr.post(path="/closed-objects-merging-sld-s3/",
                                     responses={200: {}, 500: {}, 503: {}},
                                     status_code=status.HTTP_200_OK,
                                     response_class=Response,
                                     tags=["Closed objects merging", "S3"],
                                     summary="Closed objects merging for SLD in S3")
async def closed_objects_merging_endpoint_s3(files_data: ClosedObjectsMergingSldS3FilesData) -> Response:
    # ________________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    s3 = S3FileOps(s3_bucket_name=files_data.s3_bucket_name)

    # --- DOWNLOAD JSON FILES FROM AWS S3 ---
    logger.info("- 1 - DOWNLOADING JSON FILES FROM AWS S3 -")

    json_data = ClosedObjectsMergingSldS3JsonData()

    for json_name, json_file_attrs in files_data.files.json_part.dict().items():
        if json_file_attrs is None:
            continue
        setattr(json_data, json_name, s3.get_json_file_data(s3_file_key=json_file_attrs["file_key"]))

    # ________________________________________________________________________________
    # --- MERGE OBJECTS ---
    logger.info("- 2 - MERGING OBJECTS -")

    merged_objects = merge_objects_sld(filtered_lines=json_data.filtered_lines,
                                       closed_objects=json_data.closed_objects,
                                       detected_objects=json_data.detected_objects,
                                       junction_boxes=json_data.junction_boxes,
                                       config=settings.SLD_OBJECTS_MERGING_CONF)

    # ________________________________________________________________________________
    # --- UPLOAD RESULTS TO AWS S3 ---
    try:
        logger.info("- 3 - UPLOADING RESULTS TO AWS S3 -")

        s3.upload_json_file_to_bucket(s3_file_key=files_data.out_files.json_part.merged_objects.file_key,
                                      data_for_json=merged_objects)

        return Response(status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"ERROR -> Failed to upload file to S3. Error: {e}")
