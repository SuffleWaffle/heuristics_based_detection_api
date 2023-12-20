import io
from zipfile import ZipFile
from typing import Tuple
import ujson

class FileProc:
    @staticmethod
    def get_zip_with_json(json_content: dict) -> Tuple[io.BytesIO, str]:
        # --- In-memory byte stream for ZIP file contents
        zip_bytes = io.BytesIO()

        with ZipFile(zip_bytes, "w") as zip_file:
            for json_name in json_content.keys():
                data_arg_bytes = ujson.dumps(json_content.get(json_name)).encode()
                zip_file.writestr(f"{json_name}.json", data_arg_bytes)

        zip_bytes.seek(0)
        resp_media_type = "application/zip"

        return zip_bytes, resp_media_type

    @staticmethod
    def get_zip_with_img(img_content: dict) -> Tuple[io.BytesIO, str]:
        # --- In-memory byte stream for ZIP file contents
        zip_bytes = io.BytesIO()

        with ZipFile(zip_bytes, "w") as zip_file:
            for img_name in img_content.keys():
                img_bytes = io.BytesIO()
                img_content.get(img_name).save(img_bytes, format="PNG")
                zip_file.writestr(f"{img_name}.png", img_bytes.getvalue())

        zip_bytes.seek(0)
        resp_media_type = "application/zip"

        return zip_bytes, resp_media_type

    @staticmethod
    def get_zip_with_json_and_img(json_content: dict, img_content: dict) -> Tuple[io.BytesIO, str]:
        # --- In-memory byte stream for ZIP file contents
        zip_bytes = io.BytesIO()

        with ZipFile(zip_bytes, "w") as zip_file:
            for json_name in json_content.keys():
                data_arg_bytes = ujson.dumps(json_content.get(json_name)).encode()
                zip_file.writestr(f"{json_name}.json", data_arg_bytes)

            for img_name in img_content.keys():
                img_bytes = io.BytesIO()
                img_content.get(img_name).save(img_bytes, format="PNG")
                zip_file.writestr(f"{img_name}.png", img_bytes.getvalue())

        zip_bytes.seek(0)
        resp_media_type = "application/zip"

        return zip_bytes, resp_media_type