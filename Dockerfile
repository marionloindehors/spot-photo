FROM python:3.10.6-buster
COPY requirements_prod.txt requirements_prod.txt
RUN pip install -r requirements_prod.txt
COPY spot_photo spot_photo
CMD uvicorn spot_photo.api.fast:app --host 0.0.0.0 --port $PORT
