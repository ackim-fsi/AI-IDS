FROM ai:python-common
MAINTAINER Mohyun Park <mhpark@fsec.or.kr>

RUN mkdir /home/dockeruser/data
RUN mkdir /home/dockeruser/npy

COPY inv_app_parse.py /home/dockeruser/inv_app_parse.py

CMD python3 inv_app_parse.py
