#/bin/bash
docker build -f inv_aut_parse.Dockerfile -t aisec:crd_inv_aut_parse .
docker build -f inv_aut_train.Dockerfile -t aisec:crd_inv_aut_train .
docker build -f inv_aut_predict.Dockerfile -t aisec:crd_inv_aut_predict .

