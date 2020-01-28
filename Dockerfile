FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update && apt-get install -y git libsm6 libxrender1 python3-dev python3.7 python3.7-dev

RUN pip install --upgrade pip Cython

ADD . .

RUN pip install -r requirements.txt

