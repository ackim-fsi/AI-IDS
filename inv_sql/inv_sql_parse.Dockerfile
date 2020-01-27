FROM ai:python-common
MAINTAINER HyeSeong Jeong <hsjeong@fsec.or.kr>

RUN mkdir /home/dockeruser/data
RUN mkdir /home/dockeruser/npy

COPY inv_sql_parse.py /home/dockeruser/inv_sql_parse.py

CMD python3 inv_sql_parse.py