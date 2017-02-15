FROM civisanalytics/datascience-python:latest

RUN apt-get install -y build-essential

COPY ./ /src/pyfm
RUN cd /src/pyfm && \
    pip install -r requirements.txt && \
    python setup.py install
