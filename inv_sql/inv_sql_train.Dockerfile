FROM ai:python-keras
MAINTAINER HyeSeong Jeong <hsjeong@fsec.or.kr>

RUN mkdir /home/dockeruser/npy
RUN mkdir /home/dockeruser/models

COPY inv_sql_train.py /home/dockeruser/train.py

CMD python3 train.py
