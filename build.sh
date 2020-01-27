#/bin/bash
docker build -f data_save.Dockerfile -t ai:data_save .
docker build -f label_save.Dockerfile -t ai:label_save .
docker build -f realtime_save.Dockerfile -t ai:realtime_save .
docker build -f data_split.Dockerfile -t ai:data_split .
docker build -f data_backup.Dockerfile -t ai:data_backup .
