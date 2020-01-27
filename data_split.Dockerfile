FROM ai:python-common
MAINTAINER Mohyun Park <mhpark@fsec.or.kr>

RUN mkdir /home/dockeruser/data
COPY data_split.py /home/dockeruser/data_split.py

CMD python3 data_split.py
