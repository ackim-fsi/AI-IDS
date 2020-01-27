FROM ai:python-common
MAINTAINER Mohyun Park <mhpark@fsec.or.kr>

RUN mkdir /home/dockeruser/data
RUN mkdir /home/dockeruser/prediction
RUN mkdir /home/dockeruser/data_backup
RUN mkdir /home/dockeruser/prediction_backup
COPY data_backup.py /home/dockeruser/data_backup.py

CMD python3 data_backup.py
