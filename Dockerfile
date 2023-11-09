FROM python:3.9-slim

WORKDIR /src

COPY requirements.txt /src/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

COPY . /src/
