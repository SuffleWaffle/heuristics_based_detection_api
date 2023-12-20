# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
from dataclasses import dataclass, field
from typing import Optional
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import APIRouter, HTTPException, Form
from fastapi import status, UploadFile, File
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
from src_utils.loading_utils import load_model
from src_utils.loading_utils import load_pdf
from src_utils.plotting_uitls import plot_grids_result
from src_utils.zipping_utils import FileProc
from src_utils.aws_utils import S3FileOps
from src_routers.request_models.pydantic_models_s3 import GridLinesDetectionFilesDataS3
# ---- PROCESSES ----
from src_processes.proc_find_grid_lines import find_grid_lines
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.INFO)

# ________________________________________________________________________________
# --- INIT CONFIG - LOAD MODEL ---
LGB_MODEL = load_model(settings.GRID_LINES_CONF["model_filtering"]["model_path"])

# ________________________________________________________________________________
# --- FastAPI ROUTER ---
grid_lines_rtr = APIRouter(prefix="/v1")


class GridLinesData(BaseModel):
    lines: dict
    cubic_lines: dict
    circles: dict
    rectangles: dict
    quads: dict
    parsed_text: dict
    grids_areas: list = []


# ________________________________________________________________________________
@grid_lines_rtr.post(path="/grid-lines-detection-zip/",
                     responses={200: {}, 500: {}, 503: {}},
                     status_code=status.HTTP_200_OK,
                     response_class=StreamingResponse,
                     tags=["Grid Lines Detection"],
                     summary="Detects grid lines and returns result in zip")
async def grid_lines_endpoint_zip(lines_file: UploadFile = File(...),
                                  cubic_lines_file: UploadFile = File(...),
                                  circles_file: UploadFile = File(...),
                                  rectangles_file: UploadFile = File(...),
                                  quads_file: UploadFile = File(...),
                                  parsed_text_file: UploadFile = File(...),
                                  grids_areas_file: Optional[UploadFile] = File(None),
                                  pdf_file: UploadFile = File(...),
                                  page_num: int = Form(0),
                                  visualize_results: bool = Form(True)
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

    if not grids_areas_file.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {grids_areas_file.filename} does not end with .json')

    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(404, f'File with filename {pdf_file.filename} does not end with .pdf')

    # load jsons
    lines = json.loads(await lines_file.read())
    cubic_lines = json.loads(await cubic_lines_file.read())
    circles = json.loads(await circles_file.read())
    rectangles = json.loads(await rectangles_file.read())
    quads = json.loads(await quads_file.read())
    grids_areas = json.loads(await grids_areas_file.read())
    parsed_text = json.loads(await parsed_text_file.read())
    # load pdf
    img_array= load_pdf(pdf_file, page_num=page_num)[2]
    # get grid lines
    grid_lines = find_grid_lines(lines=lines,
                                 cubic_lines=cubic_lines,
                                 circles=circles,
                                 rectangles=rectangles,
                                 quads=quads,
                                 parsed_text=parsed_text,
                                 config=settings.GRID_LINES_CONF,
                                 model=LGB_MODEL,
                                 grids_areas=grids_areas)

    if visualize_results:
        img_array = plot_grids_result(img_array, grid_lines)
        json_content = {"grid_lines": grid_lines}
        img_content = {"test_img": img_array}
        resp_content, resp_media_type = FileProc.get_zip_with_json_and_img(json_content=json_content,
                                                                           img_content=img_content)

    else:
        json_content = {"grid_lines": grid_lines}
        resp_content, resp_media_type = FileProc.get_zip_with_json(json_content=json_content)

    return StreamingResponse(content=resp_content,
                             status_code=status.HTTP_200_OK,
                             media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=grid_lines.zip"})


# ________________________________________________________________________________
@grid_lines_rtr.post(path="/grid-lines-detection-json/",
                     responses={200: {}, 500: {}, 503: {}},
                     status_code=status.HTTP_200_OK,
                     response_class=Response,
                     tags=["Grid Lines Detection"],
                     summary="Detects grid lines and returns result in json")
async def grid_lines_endpoint_json(item: GridLinesData) -> UJSONResponse:
    grid_lines = find_grid_lines(**item.__dict__,
                                 config=settings.GRID_LINES_CONF,
                                 model=LGB_MODEL)

    return UJSONResponse(content=grid_lines)


@dataclass(slots=True)
class GridLinesDetectionJsonData:
    lines:          dict = None
    cubic_lines:    dict = None
    circles:        dict = None
    rectangles:     dict = None
    quads:          dict = None
    parsed_text:    dict = None
    grids_areas:    list = field(default_factory=list)


# ________________________________________________________________________________
@grid_lines_rtr.post(path="/grid-lines-detection-json-s3/",
                     responses={200: {}, 500: {}, 503: {}},
                     status_code=status.HTTP_200_OK,
                     response_class=Response,
                     tags=["Grid Lines Detection", "S3"],
                     summary="Detects grid lines and returns result in json")
async def grid_lines_endpoint_json_s3(files_data: GridLinesDetectionFilesDataS3) -> Response:
    # ________________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    s3 = S3FileOps(s3_bucket_name=files_data.s3_bucket_name)

    # --- INIT DATA CONTAINER INSTANCE of GridLinesDetectionJsonData dataclass ---
    json_data = GridLinesDetectionJsonData()

    # --- ITERATE OVER JSON-FILES KEYS (attributes of | GridLinesDetectionFiles | Pydantic model).
    # --- "json_data" is a dataclass of dicts with multiple JSONs data payloads that
    # --- are accessible by attribute names equivalent to fields in respective Pydantic model ---
    for attr_name, json_file_attrs in files_data.files.dict().items():
        if json_file_attrs is None:
            continue
        setattr(json_data, attr_name, s3.get_json_file_data(s3_file_key=json_file_attrs["file_key"]))

    # ________________________________________________________________________________
    # --- GET GRID LINES ---
    grid_lines = find_grid_lines(lines=json_data.lines,
                                 cubic_lines=json_data.cubic_lines,
                                 circles=json_data.circles,
                                 rectangles=json_data.rectangles,
                                 quads=json_data.quads,
                                 parsed_text=json_data.parsed_text,
                                 grids_areas=json_data.grids_areas,
                                 config=settings.GRID_LINES_CONF,
                                 model=LGB_MODEL)

    # ________________________________________________________________________________
    # --- UPLOAD RESULTS TO AWS S3 BUCKET ---
    try:
        logger.info("- 4.1 - STARTED UPLOADING RESULTS TO AWS S3 -")

        s3.upload_json_file_to_bucket(s3_file_key=files_data.out_s3_file_key,
                                      data_for_json=grid_lines)

        logger.info("- 4.2 - UPLOADED JSON FILE TO AWS S3 -")

        return Response(status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"ERROR -> Raised EXCEPTION during failed upload of the file to S3. \n")
