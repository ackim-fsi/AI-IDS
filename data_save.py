# -*- coding: utf-8 -*-
import os
import datetime
import sys
import shutil


def append_data(data_dir, data_type, splunk_query, credentials, headers, data_count=0, sample_ratio=1):

    yesterday = datetime.datetime.today() + datetime.timedelta(days=-1)

    yesterday_string = yesterday.strftime("%Y%m%d")
    new_filename = data_dir + yesterday_string + "_" + data_type + ".csvx"

    if not os.path.exists(new_filename):
        os.umask(0)
        with open(os.open(new_filename, os.O_CREAT | os.O_WRONLY, 0o666), 'a') as f:
            f.write(','.join(headers) + '\n')

    # 전일 데이터 입력
    splunk_data = None
    while splunk_data is None:
        splunk_data = fsi_splunk.query(
            "http://192.168.143.39:8089",
            splunk_query,
            check_frequency=10,
            output_format='csv',
            auth=credentials,
            sample_ratio=sample_ratio,
            output_count=data_count
        )

    if len(splunk_data) > 0:
        with open(new_filename,
                  mode='a',
                  encoding='utf-8',
                  newline='') as f:
            splunk_data = splunk_data[splunk_data.index('\n'):]
            f.write(splunk_data + "\n")


def replace_data(data_path, splunk_query, credentials, headers, data_count=0, sample_ratio=1):

    splunk_data = None
    while splunk_data is None:
        splunk_data = fsi_splunk.query(
            "http://192.168.143.39:8089",
            splunk_query,
            check_frequency=10,
            output_format='csv',
            auth=credentials,
            output_count=data_count
        )

    with open(data_path,
              mode='w',
              encoding='utf-8',
              newline='') as f:
        f.write(splunk_data + '\n')


if __name__ == "__main__":
    import fsi_splunk
    from splunk_queries import search_query_total, search_query_label, search_query_payload

    credentials = ('airesearch', 'airflow!@')

    if sys.argv[-1] == "payload":

        payload_headers = [
            "_time",
            "src_ip",
            "src_port",
            "dest_ip",
            "dest_port",
            "src_content",
            "TOP_CATE_NM",
            "MID_CATE_NM",
            "suppression",
            "desc",
            "drill",
            "msg",
            "label"
        ]

        append_data(
            "./data/",
            "total",
            search_query_total(headers=payload_headers),
            credentials=credentials,
            headers=payload_headers,
            data_count=500000,
        )

    elif sys.argv[-1] == "label":

        label_headers = [
            "_time",
            "src_ip",
            "src_port",
            "dest_ip",
            "dest_port",
            "src_content",
            "model_name",
            "label",
            "comment"
        ]

        append_data(
            "./data/",
            "label",
            search_query_label(earliest_minute=-1500, latest_minute=-1440, headers=label_headers),
            credentials=credentials,
            headers=label_headers,
            data_count=100000
        )

    elif sys.argv[-1] == "realtime":

        realtime_headers = [
            "_time",
            "tas",
            "src_ip",
            "src_port",
            "dest_ip",
            "dest_port",
            "src_content",
        ]

        data_dir = "./data/"
        holder_filename = "payload_holder.tmp"
        payload_filename = "payload_data.tmp"

        holder_path = os.path.join(data_dir, holder_filename)
        payload_path = os.path.join(data_dir, payload_filename)

        replace_data(
            holder_path,
            search_query_payload(earliest_minute=-200, latest_minute=-180, headers=realtime_headers),
            credentials=credentials,
            headers=realtime_headers,
            data_count=1000000
        )
        shutil.copy(holder_path, payload_path)
