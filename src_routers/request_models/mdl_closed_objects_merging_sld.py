# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
from typing import Optional
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from pydantic import BaseModel
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
from .mdl_general import PDFFileAttrs, JSONFileAttrs
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- ClosedObjectsMergingSldS3FilesData - DATA MODELS FOR __INPUT__ FILES ---
class ClosedObjectsMergingSldS3FilesJSON(BaseModel):
    filtered_lines:     JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    closed_objects:     JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    detected_objects:   JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    junction_boxes:     JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")


class ClosedObjectsMergingSldS3Files(BaseModel):
    json_part: ClosedObjectsMergingSldS3FilesJSON


# ________________________________________________________________________________
# --- ClosedObjectsMergingSldS3FilesData - DATA MODELS FOR __OUTPUT__ FILES ---
class ClosedObjectsMergingSldS3OutFilesJSON(BaseModel):
    merged_objects:     JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")


class ClosedObjectsMergingSldS3OutFiles(BaseModel):
    json_part: ClosedObjectsMergingSldS3OutFilesJSON


# ________________________________________________________________________________
# --- ClosedObjectsMergingSldS3FilesData - MAIN DATA MODEL ---
class ClosedObjectsMergingSldS3FilesData(BaseModel):
    """
    - Data model for the request body of the "/closed-objects-merging-sld-s3/" endpoint.
    """
    files: ClosedObjectsMergingSldS3Files
    out_files: ClosedObjectsMergingSldS3OutFiles

    s3_bucket_name: str = "drawer-ai-services-test-files"

    class Config:
        title = "ClosedObjectsMergingSldS3FilesData"
        schema_extra = {
            "files": {
                "json_part": {
                    "filtered_lines": {
                        "file_key": "sample/file/path.pdf"
                    },
                    "closed_objects": {
                        "file_key": "sample/file/path.json"
                    },
                    "detected_objects": {
                        "file_key": "sample/file/path.json"
                    },
                    "junction_boxes": {
                        "file_key": "sample/file/path.json"
                    }
                }
            },
            "out_files": {
                "json_part": {
                    "merged_objects": {
                        "file_key": "sample/file/path.json"
                    }
                }
            },
            "s3_bucket_name": "drawer-ai-services-test-files",
        }
