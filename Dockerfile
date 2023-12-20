# --- Base Debian 12 Bookworm (Ubuntu-family) Python 3.11.4 image
FROM python:3.11.4-bookworm

# --- Install basic required Linux packages
RUN apt-get clean && apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common build-essential && \
    apt-get update --fix-missing

# --- Install Linux packages required for Python and OpenCV
RUN apt-get clean && apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-dev python3-pip && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get update --fix-missing

# --- SET THE WORKING DIRECTORY ---
WORKDIR /app

# --- Install Python dependencies and packages
COPY requirements.txt /app
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# --- Clean up the system from unused packages
RUN apt-get autoremove --purge -y build-essential gfortran libatlas-base-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Copy the APP code to the source
COPY . /app


# - Set default ENV variables
#ENV APP_HOST=0.0.0.0
#ENV APP_PORT=8024
#ENV APP_WORKERS=1
#ENV APP_TIMEOUT_SEC=1200
#ENV APP_GRACEFUL_TO_SEC=120

# - Expose API port
#EXPOSE $APP_PORT/tcp

# - Gunicorn PROD command to run FastAPI APP
#CMD gunicorn main:app --worker-tmp-dir /dev/shm -b $APP_HOST:$APP_PORT -w $APP_WORKERS -t $APP_TIMEOUT_SEC --graceful-timeout $APP_GRACEFUL_TO_SEC -k uvicorn.workers.UvicornWorker