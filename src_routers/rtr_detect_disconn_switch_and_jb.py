# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
# import io
from dataclasses import dataclass
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import APIRouter, HTTPException, Form, File, UploadFile, status
from fastapi.responses import UJSONResponse, Response
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import ujson as json
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
# ---- CONFIG ----
import settings
from src_logging import log_config
# ---- UTILS ----
from src_utils.loading_utils import load_pdf
from src_utils.plotting_uitls import plot_jb_ds_result
from src_utils.zipping_utils import FileProc
from src_utils.aws_utils import S3FileOps
# ---- PROCESSES ----
from src_processes.disconnected_switch_nd_text_heuristic import \
    detect_by_text_n_disconnected_heuristic
# ---- REQUEST MODELS ----
from src_routers.request_models.pydantic_models_s3 import DetectJunctionBoxesFilesDataS3
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.INFO)

# ________________________________________________________________________________
# --- FastAPI ROUTER ---
ds_jb_rtr = APIRouter(prefix="/v1")


class DSJBData(BaseModel):
    lines: dict
    cubic_lines: dict
    circles: dict
    rectangles: dict
    quads: dict
    parsed_text: dict


# ________________________________________________________________________________
@ds_jb_rtr.post(path="/detect-jb-zip/",
                responses={200: {}, 500: {}, 503: {}},
                status_code=status.HTTP_200_OK,
                response_class=StreamingResponse,
                tags=["Disconnected Switches and Junction Boxes Detection", "ZIP"],
                summary="Detect disconnected switches, junction boxes and return zip")
async def ds_jb_endpoint_zip(lines_file: UploadFile = File(...),
                             cubic_lines_file: UploadFile = File(...),
                             circles_file: UploadFile = File(...),
                             rectangles_file: UploadFile = File(...),
                             quads_file: UploadFile = File(...),
                             parsed_text_file: UploadFile = File(...),
                             pdf_file: UploadFile = File(...),
                             page_num: int = Form(0),
                             visualize_results: bool = Form(True),
                             to_return_lines: bool = Form(False)
                             ) -> StreamingResponse:
    if not lines_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {lines_file.filename} does not end with .json')

    if not cubic_lines_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {cubic_lines_file.filename} does not end with .json')

    if not circles_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {circles_file.filename} does not end with .json')

    if not rectangles_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {rectangles_file.filename} does not end with .json')

    if not quads_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {quads_file.filename} does not end with .json')

    if not parsed_text_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {parsed_text_file.filename} does not end with .json')

    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(404, f'File with filename {pdf_file.filename} does not end with .pdf')

    # load jsons
    lines = json.loads(await lines_file.read())
    cubic_lines = json.loads(await cubic_lines_file.read())
    circles = json.loads(await circles_file.read())
    rectangles = json.loads(await rectangles_file.read())
    quads = json.loads(await quads_file.read())
    parsed_text = json.loads(await parsed_text_file.read())
    # load pdf
    img_array= load_pdf(pdf_file, page_num=page_num)[2]
    # classify
    classified = detect_by_text_n_disconnected_heuristic(lines=lines,
                                                         cubic_lines=cubic_lines,
                                                         circles=circles,
                                                         rectangles=rectangles,
                                                         quads=quads,
                                                         parsed_text=parsed_text,
                                                         config=settings.JUNCTION_BOX_DETECTION_CONF,
                                                         to_return_lines=to_return_lines)

    if visualize_results:
        img_array = plot_jb_ds_result(img_array, classified['data'])
        json_content = {"detected_jb_ds": classified}
        img_content = {"test_img": img_array}
        resp_content, resp_media_type = FileProc.get_zip_with_json_and_img(json_content=json_content,
                                                                           img_content=img_content)

    else:
        json_content = {"detected_jb_ds": classified}
        resp_content, resp_media_type = FileProc.get_zip_with_json(json_content=json_content)


    return StreamingResponse(content=resp_content,
                             status_code=status.HTTP_200_OK,
                             media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=grid_lines.zip"})


# ________________________________________________________________________________
@ds_jb_rtr.post(path="/detect-jb-json/",
                responses={200: {}, 500: {}, 503: {}},
                status_code=status.HTTP_200_OK,
                response_class=Response,
                tags=["Disconnected Switches and Junction Boxes Detection"],
                summary="Detect disconnected switches, junction boxes and return json")
async def ds_jb_endpoint_json(item: DSJBData,
                              to_return_lines=False) -> UJSONResponse:
    classified = detect_by_text_n_disconnected_heuristic(**item.__dict__,
                                                         config=settings.JUNCTION_BOX_DETECTION_CONF,
                                                         to_return_lines=to_return_lines)

    return UJSONResponse(content=classified)


@dataclass(slots=True)
class DetectJunctionBoxesJsonData:
    lines:          dict = None
    cubic_lines:    dict = None
    circles:        dict = None
    rectangles:     dict = None
    quads:          dict = None
    parsed_text:    dict = None


# ________________________________________________________________________________
@ds_jb_rtr.post(path="/detect-jb-json-s3/",
                responses={200: {}, 500: {}, 503: {}},
                status_code=status.HTTP_200_OK,
                response_class=Response,
                tags=["Disconnected Switches and Junction Boxes Detection", "S3"],
                summary="Detect disconnected switches, junction boxes and return json")
async def ds_jb_endpoint_json_s3(files_data: DetectJunctionBoxesFilesDataS3) -> Response:
    # ________________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    s3 = S3FileOps(s3_bucket_name=files_data.s3_bucket_name)

    # --- ITERATE OVER JSON-FILES KEYS (attributes of | DetectJunctionBoxesFiles | Pydantic model).
    # --- "json_data" is a Dict of Dicts with multiple JSONs data payloads that
    # --- are accessible by keys equivalent to fields in respective Pydantic model ---
    logger.info("- 1.1 - STARTED DOWNLOADING JSON FILES FROM AWS S3 -")
    json_data = {}
    for attr_name, json_file_attrs in files_data.files.dict().items():
        json_data[attr_name] = s3.get_json_file_data(s3_file_key=json_file_attrs["file_key"])
    logger.info("- 1.2 - DOWNLOADED JSON FILES FROM AWS S3 -")

    # ________________________________________________________________________________
    # --- DETECT DISCONNECTED SWITCHES AND JUNCTION BOXES ---
    logger.info("- 2.1 - STARTED DETECTING DISCONNECTED SWITCHES AND JUNCTION BOXES -")
    classified = detect_by_text_n_disconnected_heuristic(**json_data,
                                                         config=settings.JUNCTION_BOX_DETECTION_CONF)
    logger.info("- 2.2 - DETECTED DISCONNECTED SWITCHES AND JUNCTION BOXES -")

    # ________________________________________________________________________________
    # --- UPLOAD FILE TO AWS S3 BUCKET ---
    try:
        logger.info("- 4.1 - STARTED UPLOADING JSON FILE TO AWS S3 -")
        s3.upload_json_file_to_bucket(s3_file_key=files_data.out_s3_file_key, data_for_json=classified)
        logger.info("- 4.2 - UPLOADED JSON FILE TO AWS S3 -")

        return Response(status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"ERROR -> Raised EXCEPTION during failed upload of the file to S3. \n")
