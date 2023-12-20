# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
# import asyncio
import io
import logging
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
import boto3
# import aioboto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from fastapi import HTTPException, status
import ujson as json
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
import settings
from src_logging import log_config
# >>>> ********************************************************************************

# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.DEBUG)


# ________________________________________________________________________________
class S3Utils:
    """
    - S3Utils class for "low-level" operations with AWS S3 storage.
    """
    def __init__(self,
                 access_key: str = settings.AWS_ACCESS_KEY,
                 secret_key: str = settings.AWS_SECRET_KEY,
                 region_name: str = settings.AWS_REGION_NAME):
        """
        Initialize S3Utils instance with AWS credentials (access_key, secret_key), bucket name, and region.

        Parameters:
            access_key (str): AWS access key ID
            secret_key (str): AWS secret access key
            region_name (str): AWS region. Defaults to 'us-east-1'.
        """
        self.s3 = boto3.client("s3",
                               aws_access_key_id=access_key,
                               aws_secret_access_key=secret_key,
                               region_name=region_name)
        # self.s3_session = aioboto3.Session(aws_access_key_id=access_key,
        #                                    aws_secret_access_key=secret_key,
        #                                    region_name=region_name)
        # self.s3_bucket_name = s3_bucket_name

    def download_file_obj(self, s3_bucket_name: str, s3_file_key: str) -> io.BytesIO:
        """
        Download a file from the S3 bucket.

        Parameters:
            s3_bucket_name (str): S3 bucket name.
            s3_file_key (str): S3 object name. Example: 'S3_FOLDER_NAME/FILE_NAME.pdf'

        Returns:
            io.BytesIO: io.BytesIO stream containing the file data.
        """
        file_byte_stream = io.BytesIO()

        try:
            logger.info(f"- Downloading file from S3 -")
            logger.info(f"- S3 bucket: {s3_bucket_name} - File key: {s3_file_key}")
            # - Download file as bytes into BytesIO stream
            self.s3.download_fileobj(s3_bucket_name,
                                     s3_file_key,
                                     file_byte_stream)

            # - Reset stream to the start for consumer
            file_byte_stream.seek(0)
            return file_byte_stream

        except ClientError:
            logger.error("ERROR -> ClientError -> Error with S3 client connection.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="ERROR -> ClientError -> Error with S3 client connection.")
        # except ClientError as ce:
        #     error_code = ce.response['Error']['Code']
        #     error_message = ce.response['Error']['Message']
        #     logger.error(f"ERROR -> ClientError -> Code: {error_code}, Message: {error_message}")
        #     raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        #                         detail=f"ERROR -> ClientError -> \nError Code: {error_code}, \nMessage: {error_message}")
        except NoCredentialsError:
            logger.error("ERROR -> NoCredentialsError -> Credentials not found.")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="ERROR -> NoCredentialsError -> Credentials not found.")
        except PartialCredentialsError:
            logger.error("ERROR -> PartialCredentialsError -> Incomplete credentials provided.")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="ERROR -> PartialCredentialsError -> Incomplete credentials provided.")
        except Exception as e:
            logger.error(f"ERROR -> Exception -> An error occurred: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Exception -> An error occurred: {e}")

    def upload_file_obj(self, file_byte_stream: io.BytesIO, s3_bucket_name: str, s3_file_key: str) -> bool:
        """
        Upload a file to the S3 bucket.

        Parameters:
            file_byte_stream (io.BytesIO): io.BytesIO stream containing the file data.
            s3_bucket_name (str): S3 bucket name.
            s3_file_key (str): S3 object name. Example: 'S3_FOLDER_NAME/FILE_NAME.pdf'

        Returns:
            bool: True if file was uploaded, else False.
        """

        try:
            # - Upload file as bytes from BytesIO stream
            self.s3.upload_fileobj(file_byte_stream,
                                   s3_bucket_name,
                                   s3_file_key)

            return True

        except ClientError:
            logger.error("ERROR -> ClientError -> Error with S3 client connection.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="ERROR -> ClientError -> Error with S3 client connection.")
        except NoCredentialsError:
            logger.error("ERROR -> NoCredentialsError -> Credentials not found.")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="ERROR -> NoCredentialsError -> Credentials not found.")
        except PartialCredentialsError:
            logger.error("ERROR -> PartialCredentialsError -> Incomplete credentials provided.")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="ERROR -> PartialCredentialsError -> Incomplete credentials provided.")
        except Exception as e:
            logger.error(f"ERROR -> Exception -> An error occurred: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"ERROR -> Exception: {e}")


# ________________________________________________________________________________
class S3FileOps(S3Utils):
    """
    - S3FileOps class for "high-level" operations with files in AWS S3 storage.
    """
    def __init__(self, s3_bucket_name: str, **kwargs):
        super().__init__(**kwargs)
        self.s3_bucket_name: str = s3_bucket_name

    def get_json_file_data(self, s3_file_key: str) -> dict:
        """
        Download JSON file from S3 bucket and return decoded JSON data.

        Parameters:
            s3_file_key (str): File key for S3 object. Example: "sample/file/path.json"

        Returns:
            dict: Decoded JSON data.
        """
        json_bytes = self.download_file_obj(s3_bucket_name=self.s3_bucket_name,
                                            s3_file_key=s3_file_key)
        json_decoded = json_bytes.getvalue().decode("utf-8")
        json_data = json.loads(json_decoded)

        return json_data

    def get_pdf_file_obj_bytes(self, s3_file_key: str) -> io.BytesIO:
        """
        Download PDF file from S3 bucket and return BytesIO stream.

        Parameters:
            s3_file_key (str): File key for S3 object. Example: "sample/file/path.pdf"

        Returns:
            io.BytesIO: BytesIO stream containing the PDF file data.
        """
        pdf_file_bytes = self.download_file_obj(s3_bucket_name=self.s3_bucket_name,
                                                s3_file_key=s3_file_key)
        return pdf_file_bytes

    def upload_json_file_to_bucket(self, s3_file_key: str, data_for_json: dict):
        """
        Upload JSON file to S3 bucket.

        Parameters:
            s3_file_key (str): File key for S3 object. Example: "sample/file/path.json"
            data_for_json (dict): JSON data to be uploaded.
        """
        # --- CONVERT JSON TO BYTE STREAM ---
        json_payload = json.dumps(data_for_json)
        json_byte_stream = io.BytesIO(json_payload.encode("utf-8"))

        # --- UPLOAD JSON FILE TO AWS S3 BUCKET ---
        upload_status = self.upload_file_obj(file_byte_stream=json_byte_stream,
                                             s3_bucket_name=self.s3_bucket_name,
                                             s3_file_key=s3_file_key)

        if not upload_status:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"ERROR -> Failed to upload JSON file to S3 bucket")
