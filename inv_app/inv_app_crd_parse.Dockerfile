FROM ai:python-common
MAINTAINER Aechan Kim <ackim@fsec.or.kr>

RUN mkdir /home/dockeruser/data
RUN mkdir /home/dockeruser/npy

COPY inv_app_crd_parse.py /home/dockeruser/inv_app_crd_parse.py

CMD python3 inv_app_crd_parse.py
