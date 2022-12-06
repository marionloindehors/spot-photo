
FROM python:3.10.6-buster
COPY spot_photo spot_photo
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -r requirements.txt
CMD uvicorn spot_photo.api.fast:app --host 0.0.0.0 --port 8000
