import os
import re
import numpy as np
import random


def available_files(folders, require_train_sample=0, require_test_sample=0):
    """
    check if there are enough samples in each folder to satisfy the requirement, if not, use what we can use
    :param folders: list of data folder for each class
    :param require_train_sample: required train samples for each class, 0 for using as much as we can
    :param require_test_sample:  required test samples for each class, 0 for using as much as we can
    :return: actually used train samples, test samples, and file lists
    """
    file_lists = []
    available_samples = []
    for folder in folders:
        file_list = os.listdir(folder)
        random.shuffle(file_list)
        file_lists.append([os.path.join(folder, x) for x in file_list])
        available_sample = len(file_list)
        available_samples.append(available_sample)
        print("available_samples: %d, folder: %s" % (available_sample, folder))
    available_sample = min(available_samples)
    train_sample = require_train_sample
    test_sample = require_test_sample
    if train_sample == 0 or test_sample == 0 or available_sample < require_train_sample + require_test_sample:
        train_sample = int(available_sample * 0.8)
        test_sample = available_sample - train_sample
        print(
            "No enough samples!! Require %d train samples and %d test samples, Get %d train samples and %d test samples." % (
                require_train_sample, require_test_sample, train_sample, test_sample))
    train_lists = [x[:train_sample] for x in file_lists]
    test_lists = [x[-test_sample:] for x in file_lists]
    return train_sample, test_sample, train_lists, test_lists


def dump_features(file_lists, output_file):
    datas = []
    l = len(file_lists)
    for i, file in enumerate(file_lists):
        datas.append(np.loadtxt(file))
        if (i % 500 == 0):
            print("%d/%d" % (i, l))
    print("%d/%d" % (i, l))
    with open(output_file, 'wb') as f:
        np.save(f, np.array(datas))
    print('done')


def load_features(file_lists, input_file):
    if not os.path.exists(input_file):
        print('initialize file %s...' % input_file)
        dump_features(file_lists, input_file)
    return np.load(input_file)


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def write_csv_file(text_dirs, labels, output_dir, file_name, max_data_num=None):
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    s = ''
    for label, path in zip(labels, text_dirs):
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            all_lines = f.readlines()
        if max_data_num is not None and len(all_lines) > max_data_num:
            all_lines = all_lines[:max_data_num]
        fix = np.array([str(label) + ',,'] * len(all_lines))
        all_lines = np.array(all_lines)
        s = s + ''.join(np.core.defchararray.add(fix, all_lines).tolist())
        # for line in all_lines:
        #     s = s + str(label) + ',,' + line
    with open(os.path.join(output_dir, file_name), 'w', encoding='utf8', errors='ignore') as f:
        f.write(re.sub(r'\bunknown\b', '<unk>', s))


def write_csv_files(train_text_dirs, test_text_dirs, train_labels, test_labels, output_csv_dir, train_file, test_file,
                    max_data_num=None, max_test_data_num=2000):
    write_csv_file(train_text_dirs, train_labels, output_csv_dir, train_file, max_data_num)
    write_csv_file(test_text_dirs, test_labels, output_csv_dir, test_file, max_test_data_num)
