FROM ai:python-common
MAINTAINER Mohyun Park <mhpark@fsec.or.kr>

RUN mkdir /home/dockeruser/data
COPY data_save.py /home/dockeruser/data_save.py
COPY fsi_splunk.py /home/dockeruser/fsi_splunk.py
COPY splunk_queries.py /home/dockeruser/splunk_queries.py

CMD python3 data_save.py payload
