# -*- coding: utf-8 -*-
import os
import numpy as np
import gc
import multiprocessing
import pandas
import datetime
import sys
from pandas.errors import EmptyDataError

from functools import partial



def convert_content(content_string, x_dim, pad_before=True):

    int_list = list(map(np.uint8, str(content_string).encode('utf-8')))[:x_dim]
    if len(int_list) < x_dim:
        if pad_before:
            int_list = [np.uint8(0)] * (x_dim - len(int_list)) + int_list  # Pad Before
        else:
            int_list = int_list + [np.uint8(0)] * (x_dim - len(int_list))  # Pad After

    return int_list


def convert_data(start_index, filename, npy_dir, batch_size, x_dim, pad_before=True, augmentation=1):

    try:
        dataframe = pandas.read_csv(filename,
                                    header=0,
                                    usecols=["src_content", "label"],
                                    skiprows=list(range(1, start_index)),
                                    nrows=batch_size,
                                    engine='python')
        labels = dataframe["label"].values.astype(np.uint8)
    except ValueError:
        dataframe = pandas.read_csv(filename,
                                    header=0,
                                    usecols=["src_content"],
                                    skiprows=list(range(1, start_index)),
                                    nrows=batch_size,
                                    engine='python')
        labels = np.array([np.uint8(0)] * dataframe.shape[0])

    labels = labels.reshape((labels.shape[0], 1))
    src_content = list(convert_content(x, x_dim=x_dim, pad_before=pad_before)
                       for x in dataframe["src_content"].values)

    src_content_aug = src_content
    labels_aug = np.concatenate(tuple([labels] * augmentation))

    for i in range(1, augmentation):
        if pad_before:
            src_content_aug = src_content_aug + list(
                [np.uint8(0)]*i + content[:-i] for content in src_content
            )
        else:
            src_content_aug = src_content_aug + list(
                content[:-i] + [np.uint8(0)] * i for content in src_content
            )

    src_content_aug = np.array(src_content_aug)
    file_no = int(start_index / batch_size)
    if pad_before:
        pad_string = '_prepad'
    else:
        pad_string = '_postpad'

    basename = os.path.basename(filename)
    file_extension_index = basename.rfind('.')
    save_basename = basename[:file_extension_index] + pad_string + '_' + str(file_no) + '.npy'
    save_filename = os.path.join(npy_dir, save_basename)
    np.save(save_filename, np.concatenate((src_content_aug, labels_aug), axis=1))
    gc.collect()

    return


def convert_file_list(datafile_list, npy_dir, x_dim=1000, pad_before=True, augmentation=1):

    processors = int(multiprocessing.cpu_count() / 1.5)
    line_per_processor = int(1048576 / augmentation) # pow(2, 20)

    for filepath in datafile_list:
        if pad_before:
            pad_string = '_prepad'
        else:
            pad_string = '_postpad'

        filename = os.path.basename(filepath)
        file_extension_index = filename.rfind('.')
        npy_filename = filename[:file_extension_index] + pad_string + "_0.npy"

        if npy_filename in os.listdir(npy_dir): # Check already parsed npy existence
            continue

        try:
            df_temp = pandas.read_csv(filepath, header=0, engine='python')
        except EmptyDataError:
            continue

        row_count = df_temp.shape[0]
        del(df_temp)
        gc.collect()

        pool = multiprocessing.Pool(processes=processors)

        split_size = int(np.ceil(row_count / line_per_processor))
        index_list = list(range(0, split_size*line_per_processor, line_per_processor))

        pool.map(partial(convert_data,
                         filename=filepath,
                         npy_dir=npy_dir,
                         batch_size=line_per_processor,
                         x_dim=x_dim,
                         pad_before=pad_before,
                         augmentation=augmentation
                         ),
                 index_list)

        pool.close()
        pool.join()
        gc.collect()


if __name__ == "__main__":

    yesterday = datetime.datetime.today() + datetime.timedelta(days=-1)
    day_before_yesterday = datetime.datetime.today() + datetime.timedelta(days=-2)
    yesterday_string = yesterday.strftime("%Y%m%d")
    day_before_yesterday_string = day_before_yesterday.strftime("%Y%m%d")
    data_dir = "./data/"
    npy_dir = "./npy/"

    payload_file_list = list(os.path.join(data_dir, f)
                             for f in os.listdir(data_dir)
                             if "payload" in f and day_before_yesterday_string in f)

    app_file_list = list(os.path.join(data_dir, f)
                         for f in os.listdir(data_dir)
                         if "app" in f and yesterday_string not in f)

    label_file_list = list(os.path.join(data_dir, f)
                           for f in os.listdir(data_dir)
                           if "INV-APP" in f and yesterday_string not in f)

    convert_file_list(payload_file_list, npy_dir, x_dim=1000, pad_before=True, augmentation=1)
    convert_file_list(app_file_list, npy_dir, x_dim=1000, pad_before=True, augmentation=20)
    convert_file_list(label_file_list, npy_dir, x_dim=1000, pad_before=True, augmentation=20)

