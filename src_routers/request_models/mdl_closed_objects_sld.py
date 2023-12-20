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
# --- ClosedObjectsSldS3FilesData - DATA MODELS FOR __INPUT__ FILES ---
class ClosedObjectsSldS3FilesPDF(BaseModel):
    pdf_file: PDFFileAttrs = PDFFileAttrs(file_key="sample/file/path.pdf")


class ClosedObjectsSldS3FilesJSON(BaseModel):
    lines:          JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    cubic_lines:    JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    circles:        JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    quads:          JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    rectangles:     JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")


class ClosedObjectsSldS3Files(BaseModel):
    pdf_part:    ClosedObjectsSldS3FilesPDF
    json_part:   ClosedObjectsSldS3FilesJSON


# ________________________________________________________________________________
# --- ClosedObjectsSldS3FilesData - DATA MODELS FOR __OUTPUT__ FILES ---
class ClosedObjectsSldS3OutFilesJSON(BaseModel):
    closed_objects: JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    lines:          JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")


class ClosedObjectsSldS3OutFiles(BaseModel):
    json_part: ClosedObjectsSldS3OutFilesJSON


# ________________________________________________________________________________
# --- ClosedObjectsSldS3FilesData - MAIN DATA MODEL ---
class ClosedObjectsSldS3FilesData(BaseModel):
    """
    - Data model for the request body of the "/closed-objects--sld-s3/" endpoint.
    """
    files:      ClosedObjectsSldS3Files
    out_files:  ClosedObjectsSldS3OutFiles

    s3_bucket_name: str = "drawer-ai-services-test-files"
    page_num: Optional[int] = 0

    class Config:
        title = "ClosedObjectsSldS3FilesData"
        schema_extra = {
            "files": {
                "pdf_part": {
                    "pdf_file": {
                        "file_key": "sample/file/path.pdf"
                    }
                },
                "json_keys": {
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
                    },
                    "dashed_objects": {
                        "file_key": "sample/file/path.json"
                    }
                }
            },
            "out_files": {
                "json_keys": {
                    "closed_objects": {
                        "file_key": "sample/file/path.json"
                    },
                    "lines": {
                        "file_key": "sample/file/path.json"
                    }
                }
            },
            "s3_bucket_name": "drawer-ai-services-test-files",
        }
