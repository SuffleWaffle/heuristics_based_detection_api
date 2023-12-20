# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from pydantic import BaseModel
# >>>> ********************************************************************************


class PDFFileAttrs(BaseModel):
    file_key: str


class JSONFileAttrs(BaseModel):
    file_key: str
