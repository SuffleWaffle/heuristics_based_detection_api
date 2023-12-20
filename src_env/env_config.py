# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import os
from pathlib import Path
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from dotenv import load_dotenv
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
# import settings
# >>>> ********************************************************************************


class EnvSetupFailedError(Exception):
    """Custom error that is raised when Environment setup fails."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def setup_env(env_file_path: Path):
    if os.path.exists(env_file_path):
        load_dotenv(dotenv_path=env_file_path)
    else:
        raise EnvSetupFailedError(message=f"ERROR: Environment setup failed -> .env file '{env_file_path}' not found.")
        # raise FileNotFoundError


def setup_env_multi_files(env_files_paths: list):
    # env_file_path = Path(settings.env_file_path)
    # env_dev_file_path = Path(settings.env_dev_file_path)
    # env_prod_file_path = Path(settings.env_prod_file_path)
    # env_files_paths: list = [env_file_path, env_dev_file_path, env_prod_file_path]

    for env_file_path in env_files_paths:
        if os.path.exists(env_file_path):
            load_dotenv(dotenv_path=env_file_path)
        else:
            raise EnvSetupFailedError(message=f"ERROR: Environment setup failed -> .env file '{env_file_path}' not found.")
            # raise FileNotFoundError
