# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
from typing import Optional
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from pydantic import BaseModel
# >>>> ********************************************************************************


# ________________________________________________________________________________


class JSONFileAttrs(BaseModel):
    file_key: str


# ________________________________________________________________________________

class GridLinesDetectionFiles(BaseModel):
    lines:         JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    cubic_lines:   JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    circles:       JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    rectangles:    JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    quads:         JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    parsed_text:   JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    grids_areas:   Optional[JSONFileAttrs] = None


class GridLinesDetectionFilesDataS3(BaseModel):
    """
    - Data model for the request body of the "/grid-lines-detection-json-s3/" endpoint.
    """
    files: GridLinesDetectionFiles
    s3_bucket_name: str = "drawer-ai-services-test-files"
    out_s3_file_key: str = "sample/file/path.json"

    class Config:
        title = "GridLinesDetectionFilesDataS3"
        schema_extra = {
            "files": {
                "lines": {
                    "file_key": "sample/file/path.json"
                },
                "cubic_lines": {
                    "file_key": "sample/file/path.json"
                },
                "circles": {
                    "file_key": "sample/file/path.json"
                },
                "rectangles": {
                    "file_key": "sample/file/path.json"
                },
                "quads": {
                    "file_key": "sample/file/path.json"
                },
                "parsed_text": {
                    "file_key": "sample/file/path.json"
                },
                "grids_areas": {
                    "file_key": "sample/file/path.json"
                }
            },
            "s3_bucket_name": "drawer-ai-services-test-files",
            "out_s3_file_key": "sample/file/path.json"
        }


# ________________________________________________________________________________
class NmsObjectRefinementFiles(BaseModel):
    first_detections:   JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    second_detections:  JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")


class NmsObjectRefinementFilesDataS3(BaseModel):
    """
    - Data model for the request body of the "/nms-objects-refinement-json-s3/" endpoint.
    """
    files: NmsObjectRefinementFiles
    s3_bucket_name: str = "drawer-ai-services-test-files"
    out_s3_file_key: str = "sample/file/path.json"

    class Config:
        title = "NmsObjectRefinementFilesDataS3"
        schema_extra = {
            "files": {
                "first_detections": {
                    "file_key": "sample/file/path.json"
                },
                "second_detections": {
                    "file_key": "sample/file/path.json"
                }
            },
            "s3_bucket_name": "drawer-ai-services-test-files",
            "out_s3_file_key": "sample/file/path.json"
        }


# ________________________________________________________________________________
class DetectJunctionBoxesFiles(BaseModel):
    lines:          JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    cubic_lines:    JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    circles:        JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    rectangles:     JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    quads:          JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")
    parsed_text:    JSONFileAttrs = JSONFileAttrs(file_key="sample/file/path.json")


class DetectJunctionBoxesFilesDataS3(BaseModel):
    """
    - Data model for the request body of the /parse-sld-table-s3/ endpoint.
    """
    files: DetectJunctionBoxesFiles
    s3_bucket_name: str = "drawer-ai-services-test-files"
    out_s3_file_key: str = "sample/file/path.json"

    class Config:
        title = "DetectJunctionBoxesFilesDataS3"
        schema_extra = {
            "files": {
                "lines": {
                    "file_key": "sample/file/path.json"
                },
                "cubic_lines": {
                    "file_key": "sample/file/path.json"
                },
                "circles": {
                    "file_key": "sample/file/path.json"
                },
                "rectangles": {
                    "file_key": "sample/file/path.json"
                },
                "quads": {
                    "file_key": "sample/file/path.json"
                },
                "parsed_text": {
                    "file_key": "sample/file/path.json"
                }
            },
            "s3_bucket_name": "drawer-ai-services-test-files",
            "out_s3_file_key": "sample/file/path.json"
        }
