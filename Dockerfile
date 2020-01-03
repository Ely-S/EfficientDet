FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update && apt-get install -y git libsm6 libxrender1

RUN pip install --upgrade pip

ADD . .

RUN pip install -r requirements.txt

