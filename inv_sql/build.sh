#/bin/bash
docker build -f inv_sql_parse.Dockerfile -t aisec:crd_inv_sql_parse .
docker build -f inv_sql_train.Dockerfile -t aisec:crd_inv_sql_train .
docker build -f inv_sql_predict.Dockerfile -t aisec:crd_inv_sql_predict .
