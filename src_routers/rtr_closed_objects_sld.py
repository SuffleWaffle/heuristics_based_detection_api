# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
import io
from dataclasses import dataclass
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import APIRouter, HTTPException, Form
from fastapi import status, UploadFile, File, Response
from fastapi.responses import UJSONResponse
from starlette.responses import StreamingResponse
import ujson as json
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
# ---- CONFIG ----
import settings
from src_logging import log_config
from src_processes.find_closed_objects_sld import find_closed_objects_sld
# ---- UTILS ----
from src_utils.loading_utils import load_pdf
from src_utils.zipping_utils import FileProc
from src_utils.plotting_uitls import plot_closed_objects_sld
from src_utils.aws_utils import S3FileOps
# ---- PROCESSES ----
# ---- REQUEST MODELS ----
from src_routers.request_models.mdl_closed_objects_sld import ClosedObjectsSldS3FilesData

# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.INFO)

# ________________________________________________________________________________
# --- FastAPI ROUTER ---
closed_objects_rtr = APIRouter(prefix="/v1")


# ________________________________________________________________________________
@closed_objects_rtr.post(path="/closed-objects-sld-zip/",
                         responses={200: {}, 500: {}, 503: {}},
                         status_code=status.HTTP_200_OK,
                         response_class=StreamingResponse,
                         tags=["Closed objects SLD", "ZIP"],
                         summary="Extract closed objects from SLD in zip")
async def closed_objects_endpoint_zip(lines_file:       UploadFile = File(...),
                                      cubic_lines_file: UploadFile = File(...),
                                      circles_file:     UploadFile = File(...),
                                      rectangles_file:  UploadFile = File(...),
                                      quads_file:       UploadFile = File(...),
                                      pdf_file:         UploadFile = File(...),
                                      page_num:         int = Form(0),
                                      visualize_results: bool = Form(True)) -> StreamingResponse:
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

    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(404, f'File with filename {pdf_file.filename} does not end with .pdf')

    # get svg files
    # load jsons
    lines = json.loads(await lines_file.read())
    cubic_lines = json.loads(await cubic_lines_file.read())
    circles = json.loads(await circles_file.read())
    rectangles = json.loads(await rectangles_file.read())
    quads = json.loads(await quads_file.read())
    # get pdf file
    img_array = load_pdf(pdf_file, page_num)[2]

    # process
    closed_objects, lines = find_closed_objects_sld(lines=lines,
                                                    cubic_lines=cubic_lines,
                                                    circles=circles,
                                                    rectangles=rectangles,
                                                    quads=quads,
                                                    img_processed=img_array,
                                                    config=settings.SLD_CLOSED_OBJECTS_CONF)

    if visualize_results:
        img_array = plot_closed_objects_sld(closed_objects=closed_objects,
                                            lines=lines)
        json_content = {"closed_objects": closed_objects,
                        "lines" : lines}
        img_content = {"test_img": img_array}
        resp_content, resp_media_type = FileProc.get_zip_with_json_and_img(json_content=json_content,
                                                                           img_content=img_content)

    else:
        json_content = {"closed_objects": closed_objects,
                        "lines": lines}
        resp_content, resp_media_type = FileProc.get_zip_with_json(json_content=json_content)

    return StreamingResponse(content=resp_content,
                             status_code=status.HTTP_200_OK,
                             media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=closed_object.zip"})


# ________________________________________________________________________________
@closed_objects_rtr.post(path="/closed-objects-sld-json/",
                         responses={200: {}, 500: {}, 503: {}},
                         status_code=status.HTTP_200_OK,
                         response_class=StreamingResponse,
                         tags=["Closed objects SLD"],
                         summary="Extract closed objects from SLD in json")
async def closed_objects_endpoint_json(lines_file: UploadFile = File(...),
                                       cubic_lines_file: UploadFile = File(...),
                                       circles_file: UploadFile = File(...),
                                       rectangles_file: UploadFile = File(...),
                                       quads_file: UploadFile = File(...),
                                       pdf_file: UploadFile = File(...),
                                       page_num: int = 0) -> UJSONResponse:
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

    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(404, f'File with filename {pdf_file.filename} does not end with .pdf')

    # get svg files
    # load jsons
    lines = json.loads(await lines_file.read())
    cubic_lines = json.loads(await cubic_lines_file.read())
    circles = json.loads(await circles_file.read())
    rectangles = json.loads(await rectangles_file.read())
    quads = json.loads(await quads_file.read())
    # get pdf file
    img_array = load_pdf(pdf_file, page_num)[2]

    # process
    closed_objects, lines = find_closed_objects_sld(lines=lines,
                                                    cubic_lines=cubic_lines,
                                                    circles=circles,
                                                    rectangles=rectangles,
                                                    quads=quads,
                                                    img_processed=img_array,
                                                    config=settings.SLD_CLOSED_OBJECTS_CONF)

    json_content = {"closed_objects": closed_objects,
                    "lines": lines}

    return UJSONResponse(content=json_content)


@dataclass(slots=True)
class ClosedObjectsProcessingSLDJsonData:
    lines:          dict = None
    cubic_lines:    dict = None
    circles:        dict = None
    quads:          dict = None
    rectangles:     dict = None


# ________________________________________________________________________________
@closed_objects_rtr.post(path="/closed-objects-sld-s3/",
                         responses={200: {}, 500: {}, 503: {}},
                         status_code=status.HTTP_200_OK,
                         response_class=Response,
                         tags=["Closed objects SLD", "S3"],
                         summary="Extract closed objects from SLD in s3")
async def closed_objects_endpoint_s3(files_data: ClosedObjectsSldS3FilesData) -> Response:
    # ________________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    s3 = S3FileOps(s3_bucket_name=files_data.s3_bucket_name)

    # --- DOWNLOAD PDF FILE ---
    logger.info("- 1.1 - DOWNLOADING PDF FILE FROM AWS S3 -")
    pdf_file_bytes = s3.get_pdf_file_obj_bytes(s3_file_key=files_data.files.pdf_part.pdf_file.file_key)

    # --- DOWNLOAD JSON FILES ---
    logger.info("- 1.2 - DOWNLOADING JSON FILES FROM AWS S3 -")

    json_data = ClosedObjectsProcessingSLDJsonData()

    # --- ITERATE OVER JSON-FILES KEYS (attributes of | ClosedObjectsSldS3FilesData | Pydantic model).
    # --- "json_data" is an instance of a ClosedObjectsSLDFJSONData dataclass with multiple JSONs data payloads that
    # --- are accessible by attribute names equivalent to fields in respective Pydantic model ---
    for attr_name, json_file_attrs in files_data.files.json_part.dict().items():
        setattr(json_data, attr_name, s3.get_json_file_data(s3_file_key=json_file_attrs["file_key"]))

    # ________________________________________________________________________________
    # --- LOADING PDF FILE DATA ---
    __, ___, img_array, pdf_size = load_pdf(pdf_file_obj=pdf_file_bytes,
                                            page_num=files_data.page_num,
                                            s3_origin=True)

    # ________________________________________________________________________________
    # --- MAIN PROCESS ---
    closed_objects, lines = find_closed_objects_sld(lines=json_data.lines,
                                                    cubic_lines=json_data.cubic_lines,
                                                    circles=json_data.circles,
                                                    rectangles=json_data.rectangles,
                                                    quads=json_data.quads,
                                                    img_processed=img_array,
                                                    config=settings.SLD_CLOSED_OBJECTS_CONF)

    # ________________________________________________________________________________
    # --- UPLOAD RESULTS TO AWS S3 ---
    try:
        logger.info("- 4 - UPLOADING RESULTS TO AWS S3 -")

        s3.upload_json_file_to_bucket(s3_file_key=files_data.out_files.json_part.closed_objects.file_key,
                                      data_for_json=closed_objects)
        s3.upload_json_file_to_bucket(s3_file_key=files_data.out_files.json_part.lines.file_key,
                                      data_for_json=lines)

        return Response(status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"ERROR -> Failed to upload file to S3. Error: {e}")
