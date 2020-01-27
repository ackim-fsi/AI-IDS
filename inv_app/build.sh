#/bin/bash
docker build -f inv_app_parse.Dockerfile -t aisec:inv_app_parse .
docker build -f inv_app_crd_train.Dockerfile -t aisec:inv_app_crd_train .