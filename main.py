# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import os
import sys
import logging
import asyncio
from pathlib import Path
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import FastAPI, Response, status
from fastapi.responses import UJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ujson as json
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
# ---- CONFIG ----
# import settings
from src_logging import log_config
from src_env import env_config
# ---- FastAPI ROUTERS ----
from src_routers.rtr_detect_disconn_switch_and_jb import ds_jb_rtr
from src_routers.rtr_nms_objects_refinement import nms_objects_refinement_rtr
from src_routers.rtr_find_grid_lines import grid_lines_rtr
from src_routers.rtr_closed_objects_sld import closed_objects_rtr
from src_routers.rtr_sld_objects_merging import closed_objects_merging_sld_rtr
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.INFO)

# ________________________________________________________________________________
# --- INIT CONFIG - ENVIRONMENT LOADING ---
env_file_path = Path(".env")
env_dev_file_path = Path("src_env/dev/.env")
env_prod_file_path = Path("src_env/prod/.env")
env_stage_file_path = Path("src_env/stage/.env")

if os.getenv("ENVIRONMENT") is None:
    logger.error(">>> ENV VAR | ENVIRONMENT | IS NOT SET - SETTING TO DEFAULT: 'DEVELOPMENT' <<<")
    os.environ["ENVIRONMENT"] = "DEVELOPMENT"

if os.getenv("ENVIRONMENT") == "PRODUCTION":
    logger.info(">>> ENV VAR | ENVIRONMENT | IS SET TO: 'PRODUCTION' <<<")
    env_config.setup_env(env_file_path=env_prod_file_path)

elif os.getenv("ENVIRONMENT") == "DEVELOPMENT":
    logger.info(">>> ENV VAR | ENVIRONMENT | IS SET TO: 'DEVELOPMENT' <<<")
    env_config.setup_env(env_file_path=env_dev_file_path)

elif os.getenv("ENVIRONMENT") == "STAGE":
    logger.info(">>> ENV VAR | ENVIRONMENT | IS SET TO: 'STAGE' <<<")
    env_config.setup_env(env_file_path=env_stage_file_path)

# ________________________________________________________________________________
# --- INIT CONFIG - EVENT LOOP POLICY SETUP ---
if sys.platform == "linux":
    # --- PROD - uvloop (for Linux) EVENT LOOP POLICY SETUP ---
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logging.info(">>>> EVENT LOOP POLICY SETUP - PROD - | uvloop.EventLoopPolicy | IS ACTIVE <<<<")

elif sys.platform == "win32":
    # --- DEV - win32 (for Windows) EVENT LOOP POLICY SETUP ---
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logging.info(">>>> EVENT LOOP POLICY SETUP - DEV - | WindowsSelectorEventLoopPolicy | IS ACTIVE <<<<")


# ________________________________________________________________________________
# >>>> </> FastAPI APP CONFIG </>
description = """
**Heuristic Based Detection API service which uses algorithms and heuristics to find objects and refine their positions**  

## Healthcheck:

- **Healthcheck allows to monitor operational status of the API -> Returns status <HTTP_200_OK> if service instance is running**)

## NMS 
- **/nms-objects-refinement-zip     ->  Merge and filtering JSON by NMS ->  Return ZIP with merged and filtered by NMS bboxes**

- **/nms-objects-refinement-json     ->  Merge and filtering JSON by NMS ->  Return JSON with merged and filtered by NMS bboxes**

## Grid lines detection 
- **/grid-lines-detection-zip     -> Find grid lines and return zip**

- **/grid-lines-detection-json     ->  Find grid lines and return json**

## Detection of junction boxes and disconnected switches
- **/detect-jb-zip -> Detect junction boxes and disconnected switches -> Return ZIP with detected objects**

- **/detect-jb-json -> Detect junction boxes and disconnected switches -> Return JSON with detected objects**


"""

app = FastAPI(
    title="Drawer AI - Heuristic Based Detection API",
    description=description,
    version="0.1.0",)

app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=json.loads(os.getenv("APP_CORS_ORIGINS")),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router=nms_objects_refinement_rtr)
app.include_router(router=grid_lines_rtr)
app.include_router(router=ds_jb_rtr)
app.include_router(router=closed_objects_rtr)
app.include_router(router=closed_objects_merging_sld_rtr)


# ________________________________________________________________________________
# >>>> </> APP - STARTUP </>
@app.on_event(event_type="startup")
async def startup_event():
    logger.info(">>> Heuristic Based Detection API - SERVICE STARTUP COMPLETE <<<")


# ________________________________________________________________________________
# >>>> </> APP - SHUTDOWN </>
@app.on_event(event_type="shutdown")
async def shutdown_event():
    logger.info(">>> Heuristic Based Detection API - SERVICE SHUTDOWN <<<")


# ________________________________________________________________________________
# >>>> </> APP - HEALTHCHECK </>
class HealthcheckResponse(BaseModel):
    healthcheck: str = "API Status 200"


@app.get(path="/healthcheck/",
         status_code=status.HTTP_200_OK,
         response_model=HealthcheckResponse,
         tags=["HEALTHCHECK"],
         summary="Healthcheck endpoint for API service.")
async def healthcheck() -> Response:
    logger.info("--- HEALTHCHECK Endpoint - Status 200 ---")
    return UJSONResponse(content={"healthcheck": "API Status 200"},
                         status_code=status.HTTP_200_OK)
