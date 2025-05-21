FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

WORKDIR /app

ADD requirements.txt /app/

RUN apt-get update \
	&& apt-get install -y python3-pip \
	&& pip install -r requirements.txt --break-system-packages --no-cache

COPY . /app

ENTRYPOINT ["python3", "/app/predict.py"]
