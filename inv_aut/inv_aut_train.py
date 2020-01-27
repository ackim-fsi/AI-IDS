import datetime
import gc
import threading
import os
import shutil
from copy import deepcopy

import keras.backend.tensorflow_backend as keras_tf_backend
import keras.utils
import numpy as np
import psutil
import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout
from keras.layers import LSTM, Dense, Bidirectional, CuDNNLSTM
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import Sequential


class DataSequence(keras.utils.Sequence):

    def _find_npy_size(self, filename):
        with open(filename, 'rb') as f:
            data = str(f.read(100)[50:])
            shape_index = data.find('shape')
            comma_index = data[shape_index:].find(',')

            return int(data[shape_index+9 : shape_index+comma_index])

    def _load_cache(self, label, idx):

        cache_end_index = self._cache_end_index_dict[label]
        cache_start_index = self._cache_start_index_dict[label]
        if (idx + 1) * self._batch_size_dict[label] <= cache_end_index:
            return

        threshold = self._cache_threshold_dict[label]
        batch_size = self._batch_size_dict[label]

        delete_count = int((idx - 64) * batch_size) - cache_start_index
        if delete_count > 0:
            self._cache_dict[label] = self._cache_dict[label][delete_count:]
            self._cache_start_index_dict[label] = self._cache_start_index_dict[label] + delete_count

        new_row_count = self._cache_dict[label].shape[0]
        dict_size = self._cache_dict[label].nbytes
        if new_row_count > 0:
            self._avg_size = float(dict_size / new_row_count)

        remaining_size = threshold - dict_size
        temp_array_filename_list = []
        while self._file_size_list_dict[label]:
            file_name = self._file_list_dict[label].pop(0)
            file_size = self._file_size_list_dict[label].pop(0)
            temp_array_filename_list.append(file_name)
            remaining_size = remaining_size - file_size * self._avg_size
            self._cache_end_index_dict[label] = self._cache_end_index_dict[label] + file_size

            if not self._file_size_list_dict[label]:
                break
            if remaining_size <= self._file_size_list_dict[label][0] * self._avg_size:
                break

        self._cache_dict[label] = np.concatenate(tuple([self._cache_dict[label]] +
                                                       list(np.load(x) for x in temp_array_filename_list)
                                                       )
                                                 )
        gc.collect()

        return

    def _initialize_objects(self):
        self._file_list_dict = deepcopy(self._backup_file_list_dict)
        self._cache_dict = dict()
        gc.collect()

        for label in self._file_list_dict:
            self._file_size_list_dict[label] = []
            self._cache_dict[label] = np.empty((0, self._x_dim+1), dtype=np.uint8)
            self._cache_start_index_dict[label] = 0
            self._cache_end_index_dict[label] = 0
            for filename in self._file_list_dict[label]:
                npy_size = self._find_npy_size(filename)
                self._file_size_list_dict[label].append(npy_size)

        for label in self._cache_dict:
            self._load_cache(label, 0)

    def __init__(self, file_list_dict, x_dim, batch_size=1024, mem_share=0.2):

        self._backup_file_list_dict = deepcopy(file_list_dict)
        self._batch_size = batch_size
        self._x_dim = x_dim
        self._avg_size = (x_dim + 1)  
        self._data_size = 0
        self._file_size_list_dict = dict()
        self._batch_size_dict = dict()
        self._cache_threshold_dict = dict()
        self._cache_start_index_dict = dict()
        self._cache_end_index_dict = dict()
        self._lock_dict = dict()

        mem = psutil.virtual_memory()
        buffer = 0.25

        total_threshold = mem.available * mem_share * (1 - buffer)

        for label in file_list_dict:
            self._batch_size_dict[label] = 0
            self._lock_dict[label] = threading.Lock()

            for filename in file_list_dict[label]:
                npy_size = self._find_npy_size(filename)
                self._batch_size_dict[label] = self._batch_size_dict[label] + npy_size
                self._data_size = self._data_size + npy_size

        for label in self._batch_size_dict:
            # _batch_size_dict is not the batch size at this point. It is the total row count of each label.
            self._cache_threshold_dict[label] = int(total_threshold * self._batch_size_dict[label] / self._data_size)

            # Dividing with total data size / label size to get the batch size of each label
            self._batch_size_dict[label] = float(self._batch_size_dict[label] * self._batch_size / self._data_size)

        self._initialize_objects()

    def __len__(self):
        return int(np.ceil(self._data_size / self._batch_size))

    def __getitem__(self, idx):

        data = np.empty((0, self._x_dim+1))
        for label in self._cache_end_index_dict:

            start_index = int(idx * self._batch_size_dict[label])
            end_index = int((idx + 1) * self._batch_size_dict[label])

            with self._lock_dict[label]:
                if end_index > self._cache_end_index_dict[label]:
                    self._load_cache(label, idx)
                cache_start_index = start_index - self._cache_start_index_dict[label]

            if cache_start_index < 0:
                cache_start_index = 0

            cache_end_index = cache_start_index + (end_index - start_index)
            data = np.concatenate((data, self._cache_dict[label][cache_start_index:cache_end_index]))

        data = data[:self._batch_size]
        np.random.shuffle(data)

        train_x = data[:, :-1].reshape(data.shape[0], self._x_dim, 1)
        train_y = data[:, [-1]].reshape(data.shape[0], 1)

        train_x = (train_x - 128.0) / -128.0

        return (train_x, train_y)

    def on_epoch_end(self):
        self._initialize_objects()


def get_session(gpu_share=0.2, threads=2):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_share)
    config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                            intra_op_parallelism_threads=threads,
                            gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def initialize_hyperparameters(parameter_dict):

    if parameter_dict is None:
        # Default Values
        parameter_tuple = (2, 12, 4, 5, 16, 8, 12, 0.1, 0.1)
    else:
        parameter_tuple = (
            parameter_dict["conv_depth"],
            parameter_dict["conv_filter"],
            parameter_dict["conv_kernel_width"],
            parameter_dict["conv_pool"],
            parameter_dict["lstm_units"],
            parameter_dict["dense_depth"],
            parameter_dict["dense_units"],
            parameter_dict["dense_dropout"],
            parameter_dict["dense_relu_alpha"]
        )
    return parameter_tuple


def create_model(input_dim, hyperparameter_dict=None):

    (conv_depth,
     conv_filter,
     conv_kernel_width,
     conv_pool,
     lstm_units,
     dense_depth,
     dense_units,
     dense_dropout,
     dense_relu_alpha
    ) = initialize_hyperparameters(hyperparameter_dict)

    model = Sequential()

    # CNN Layer
    for i in range(conv_depth):
        conv_filter_size = conv_filter * pow(conv_pool, i)
        if i == 0:
            model.add(Conv1D(conv_filter_size,
                             conv_kernel_width,
                             padding='same',
                             activation='relu',
                             input_shape=(input_dim, 1)))
        else:
            model.add(Conv1D(conv_filter_size,
                             conv_kernel_width,
                             padding='same',
                             activation='relu'))
        model.add(MaxPooling1D(pool_size=conv_pool, padding='same'))
        model.add(BatchNormalization())

    # RNN Layer
    if conv_depth > 0:
        (_, lstm_timesteps, lstm_features) = model.output_shape
        lstm_input_shape = (lstm_timesteps, lstm_features) # Get input from CNN
    else:
        lstm_input_shape = (input_dim, 1) # Starts with RNN

    #model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=lstm_input_shape))
    #model.add(Bidirectional(LSTM(lstm_units)))
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=lstm_input_shape))
    #model.add(CuDNNLSTM(lstm_units, return_sequences=True, input_shape=lstm_input_shape))
    #model.add(CuDNNLSTM(lstm_units, return_sequences=True))
    #model.add(CuDNNLSTM(lstm_units))
    #model.add(CuDNNLSTM(lstm_units, input_shape=lstm_input_shape))
    #model.add(Bidirectional(CuDNNLSTM(lstm_units), input_shape=lstm_input_shape))
    model.add(Bidirectional(LSTM(lstm_units)))
    #model.add(Bidirectional(CuDNNLSTM(lstm_units)))

    # DNN Layer
    for _ in range(dense_depth):
        model.add(Dense(dense_units))
        model.add(Dropout(dense_dropout))
        model.add(LeakyReLU(dense_relu_alpha))
        model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    # print(model.summary())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, datafile_dict, x_dim, save_model=False, model_name='default.h5', verbose=0):

    processors = int(psutil.cpu_count() / 1.5)
    generator = DataSequence(datafile_dict, x_dim=x_dim, batch_size=8192, mem_share=0.2)
    model.fit_generator(generator=generator,
                        epochs=5,
                        verbose=verbose,
                        shuffle=False,
                        workers=processors
                        )

    if save_model:
        model.save(model_name)


def test_model(model, test_file_list, x_dim):

    data = np.concatenate(tuple(
        list(np.load(filename) for filename in test_file_list)
    ))

    test_size = data.shape[0]

    x_test = np.array(data[:, :-1])
    x_test = (x_test - 128.0) / -128.0
    x_test = x_test.reshape(test_size, x_dim, 1) # 1000 characters at a time, 1 channel
    y_test = data[:, [-1]].reshape(test_size, 1)
    y_prediction = model.predict(x=x_test,
                                 batch_size=4096,
                                 verbose=0)

    y_merged = (y_prediction.round()*2 + y_test).flatten()
    value, counts = np.unique(y_merged, return_counts=True)
    value_str = list(map(lambda x: str(int(x)), value))
    metrics = dict(zip(value_str, counts))

    for y_merged in ['0', '1', '2', '3']:
        if y_merged not in metrics:
            metrics[y_merged] = 0

    metrics['TP'] = metrics['3']
    metrics['FP'] = metrics['2'] # prediction is 1, label is 0 (1*2 + 0 = 2)
    metrics['FN'] = metrics['1']
    metrics['TN'] = metrics['0']
    metrics['Precision'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    metrics['Recall'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['F-Score'] = 2 * metrics['Precision'] * metrics['Recall'] /\
                         (metrics['Precision'] + metrics['Recall'])
    metrics['Accuracy'] = (metrics['TP'] + metrics['TN']) / \
                          (metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN'])

    y_test = tf.convert_to_tensor(y_test, np.float32)  # Keras Bug
    y_prediction = tf.convert_to_tensor(y_prediction, np.float32) # Keras Bug
    loss = keras.backend.eval(keras.losses.binary_crossentropy(y_true=y_test, y_pred=y_prediction))
    metrics['Loss'] = np.average(loss)

    return metrics


def search_model(datafile_dict, test_file_list, x_dim):

    if not test_file_list:
        print('Test File List is Empty!')
        return

    with open('results.csvx', 'w') as result_file:
        headers = 'conv_depth,conv_filter,conv_kernel_width,conv_pool,lstm_units,'
        headers = headers + 'dense_depth,dense_units,dense_dropout,dense_relu_alpha,'
        headers = headers + 'loss,accuracy,F-score'
        result_file.write(headers + '\n')

        for conv_depth in [1, 2, 3]:
            for conv_filter in [4, 6]:
                for conv_kernel_width in [4, 3]:
                    for conv_pool in [2, 3]:
                        for lstm_units in [16, 8]:
                            for dense_depth in [1]:
                                for dense_units in [16, 8]:
                                    for dense_dropout in [0.1, 0.3]:
                                        for dense_relu_alpha in [0.1]:
                                            hyper_p_dict = {
                                                "conv_depth" : conv_depth,
                                                "conv_filter" : conv_filter,
                                                "conv_kernel_width" : conv_kernel_width,
                                                "conv_pool" : conv_pool,
                                                "lstm_units" : lstm_units,
                                                "dense_depth" : dense_depth,
                                                "dense_units" : dense_units,
                                                "dense_dropout" : dense_dropout,
                                                "dense_relu_alpha" : dense_relu_alpha
                                            }
                                            if np.random.random() < 0.3:
                                                continue
                                            result_file.write(','.join(map(str, hyper_p_dict.values())))
                                            result_file.flush()
                                            model = create_model(x_dim, hyper_p_dict)
                                            train_model(model,
                                                        datafile_dict,
                                                        x_dim=x_dim,
                                                        save_model=False,
                                                        verbose=0)
                                            metrics = test_model(model, test_file_list, x_dim=x_dim)
                                            result_file.write(',' + str(metrics['Loss']) +
                                                              ',' + str(metrics['Accuracy']) +
                                                              ',' + str(metrics['F-Score']) + '\n')
                                            result_file.flush()
                                            keras_tf_backend.clear_session()
                                            del(model)
                                            processors = int(psutil.cpu_count() / 1.5)
                                            keras_tf_backend.set_session(get_session(0.5, processors))
                                            gc.collect()


if __name__ == '__main__':

    processors = int(psutil.cpu_count() / 1.5)
    keras_tf_backend.set_session(get_session(0.5, processors))

    yesterday = datetime.datetime.today() + datetime.timedelta(days=-1)
    day_before_yesterday = datetime.datetime.today() + datetime.timedelta(days=-2)
    yesterday_string = yesterday.strftime("%Y%m%d")
    day_before_yesterday_string = day_before_yesterday.strftime("%Y%m%d")
    npy_dir = "./npy/"
    model_dir = "./models/"
    pad_string = "prepad"

    model_category = "INV-AUT"
    model_type = "CRDNN-" + pad_string
    model_name = model_category + "-" + model_type
    model_filename = model_name + ".h5"

    model_path = os.path.join(model_dir, model_filename)
    model_backup_filename = model_filename + "." + yesterday_string
    model_backup_path = os.path.join(model_dir, model_backup_filename)

    model = create_model(1000)

    payload_files = list(os.path.join(npy_dir, f)
                         for f in os.listdir(npy_dir)
                         if "_payload_" + pad_string in f
                         and day_before_yesterday_string in f)

    aut_files = list(os.path.join(npy_dir, f)
                     for f in os.listdir(npy_dir)
                     if "_aut_" + pad_string in f
                     and yesterday_string not in f)

    label_files = list(os.path.join(npy_dir, f)
                       for f in os.listdir(npy_dir)
                       if model_name + "_" + pad_string in f)

    if os.path.exists(model_path):
        shutil.copy(model_path, model_backup_path)

    train_model(model, {'payload': payload_files, 'aut': aut_files, 'label': label_files},
                x_dim=1000, save_model=True, model_name=model_path, verbose=0)

