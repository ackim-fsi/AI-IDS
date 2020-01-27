import os
import pandas as pd
import datetime


def classify_payload(datafile_list, category_list):

    category_dict = {
        "payload" : "",
        "app" : "응용프로그램 취약점 공격",
        "aut" : "패스워드 추측 및 인증우회 공격",
        "sql" : "SQL Injection 공격"
    }

    for filename in datafile_list:
        dirname = os.path.dirname(filename)
        ymd = str(os.path.basename(filename).split('_')[0])
        df = pd.read_csv(filename, header=0)
        df = df.fillna(value={"MID_CATE_NM": ""})
        for category in category_list:
            new_filename = os.path.join(dirname, ymd + "_" + category + ".csvx")
            df[df["MID_CATE_NM"]==category_dict[category]].to_csv(
                new_filename,
                index=False,
                encoding="utf-8"
            )
            os.umask(0)
            os.chmod(new_filename, 0o666)
            os.umask(0o027)


def classify_label(labelfile_list):
    for filename in labelfile_list:
        dirname = os.path.dirname(filename)
        ymd = str(os.path.basename(filename).split('_')[0])
        df = pd.read_csv(filename, header=0)
        for model_name in df.model_name.unique():
            new_filename = os.path.join(dirname, ymd + "_" + model_name + ".csvx")
            df[df["model_name"]==model_name].to_csv(
                new_filename,
                index=False,
                encoding="utf-8"
            )
            os.umask(0)
            os.chmod(new_filename, 0o666)
            os.umask(0o027)


if __name__ == "__main__":

    mid_category_list = ["payload", "app", "aut", "sql"]
    day_before_yesterday = datetime.datetime.today() + datetime.timedelta(days=-2)
    day_string = day_before_yesterday.strftime("%Y%m%d")
    data_dir = "./data"
    file_list = list(os.path.join(data_dir, f)
                     for f in os.listdir(data_dir)
                     if day_string + "_total" in f)

    label_list = list(os.path.join(data_dir, f)
                     for f in os.listdir(data_dir)
                     if day_string + "_label" in f)

    classify_payload(file_list, mid_category_list)
    classify_label(label_list)
