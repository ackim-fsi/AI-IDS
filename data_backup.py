# -*- coding: utf-8 -*-
import os
import shutil
import datetime


def backup_data(data_dir, backup_dir, include_string, keep_days=7):

    past_day = datetime.datetime.today() + datetime.timedelta(days=-keep_days)
    past_day_string = past_day.strftime("%Y%m%d")

    for filename in os.listdir(data_dir):
        if past_day_string in filename and include_string in filename:
            from_path = os.path.join(data_dir, filename)
            to_path = os.path.join(backup_dir, filename)
            shutil.move(from_path, to_path)


if __name__ == "__main__":
    backup_data("./data", "./data_backup", "total", keep_days=3)
    backup_data("./data", "./data_backup", "payload", keep_days=3)
    backup_data("./data", "./data_backup", "aut", keep_days=28)
    backup_data("./data", "./data_backup", "app", keep_days=28)
    backup_data("./data", "./data_backup", "sql", keep_days=28)
    backup_data("./prediction", "./prediction_backup", "", keep_days=7)