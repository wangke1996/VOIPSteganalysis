import tensorflow as tf
import argparse
import os
import pickle as pkl
import time
import numpy as np
from model.biRNN1 import biLSTM
from data_helper import available_files, load_features, batch_iter
from sklearn import metrics
from sklearn.utils import shuffle

seed = 2333
np.random.seed(seed)
tf.set_random_seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


# os.environ['CUDA_VISIBLE_DEVICES']='1'
def train(train_x, train_y, test_x, test_y, args, kernal_initilizer):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # with tf.Session() as sess:
        tf.set_random_seed(seed)
        BATCH_SIZE = args.batch_size
        NUM_EPOCHS = args.num_epochs
        model = biLSTM(max_input_length=args.max_input_len, num_class=len(args.hidden_ratio), input_dim=args.input_dim,
                       hidden_layer_num=args.hidden_layers, bi_direction=args.bi_directional,
                       use_attention=args.use_attention, attention_size=args.attention_size, num_hidden=args.num_hidden,
                       fc_num_hidden=args.fc_num_hidden, hidden_layer_num_bi=args.hidden_layers_bi,
                       num_hidden_bi=args.num_hidden_bi, kernel_initializer=kernal_initilizer)
        # total_parameters = 0
        # for variable in tf.trainable_variables():
        #     # shape is an array of tf.Dimension
        #     shape = variable.get_shape()
        #     print(shape)
        #     # print(len(shape))
        #     variable_parameters = 1
        #     for dim in shape:
        #         # print(dim)
        #         variable_parameters *= dim.value
        #     # print(variable_parameters)
        #     total_parameters += variable_parameters
        # print('total parameters: %d ' % total_parameters)

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(args.lr)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)
        # Checkpoint
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load variables from pre-trained model
        if not args.pre_trained == "none":
            pre_trained_variables = [v for v in tf.global_variables() if "Adam" not in v.name]
            saver_pre = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state(args.summary_dir)
            saver_pre.restore(sess, ckpt.model_checkpoint_path)

        def print_log(s):
            with open(os.path.join(args.summary_dir, "accuracy.txt"), "a") as f:
                print(s, file=f)
            print(s)
            return

        def train_step(batch_x, batch_y):
            feed_dict = {
                model.x: batch_x,
                model.y: batch_y,
                model.keep_prob: args.keep_prob  # 0.5
            }

            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            # if step % 100 == 0:
            #     print_log("step {0} : loss = {1}".format(step, loss))
            return loss, step

        def prediction(x, y):
            batches = batch_iter(x, y, BATCH_SIZE, 1)
            outputs = []
            predictions = []
            logits = []
            for batch_x, batch_y in batches:
                logit, prediction = sess.run([model.logits, model.predictions],
                                             feed_dict={model.x: batch_x, model.y: batch_y,
                                                        model.keep_prob: 1.0})
                logits.extend(logit)
                predictions.extend(prediction.tolist())
                outputs.extend(batch_y.tolist())
            return logits, predictions, outputs

        def train_accuracy():
            _, predictions, ouputs = prediction(train_x, train_y)
            return sum(np.equal(predictions, ouputs)) / len(ouputs)

        def test_accuracy(test_x, test_y):
            _, predictions, outputs = prediction(test_x, test_y)
            labels = np.unique(outputs)
            labels_count_TP = np.array([np.sum(b.astype(int)) for b in
                                        [np.logical_and(np.equal(outputs, label_x), np.equal(predictions, label_x)) for
                                         label_x in labels]])
            labels_count_TN = np.array([np.sum(b.astype(int)) for b in [
                np.logical_not(np.logical_or(np.equal(outputs, label_x), np.equal(predictions, label_x))) for label_x in
                labels]])
            labels_count_FP = np.array([np.sum(b.astype(int)) for b in [
                np.logical_and(np.logical_not(np.equal(outputs, label_x)), np.equal(predictions, label_x)) for label_x
                in
                labels]])
            labels_count_FN = np.array([np.sum(b.astype(int)) for b in [
                np.logical_and(np.equal(outputs, label_x), np.logical_not(np.equal(predictions, label_x))) for label_x
                in
                labels]])
            precisions = labels_count_TP / (labels_count_TP + labels_count_FP)
            recalls = labels_count_TP / (labels_count_TP + labels_count_FN)
            fscores = 2 * precisions * recalls / (precisions + recalls)
            accuracies = (labels_count_TP + labels_count_TN) / (
                labels_count_TP + labels_count_TN + labels_count_FP + labels_count_FN)
            specificities = labels_count_TN / (labels_count_TN + labels_count_FP)
            all_accuracy = np.sum(labels_count_TP) / len(outputs)

            # with open(os.path.join(args.summary_dir, "accuracy.txt"), "a") as f:
            #     print("step %d: test_accuracy=%f"%(step,sum_accuracy / cnt), file=f)

            return precisions, recalls, fscores, accuracies, specificities, all_accuracy, outputs, predictions

        def write_accuracy(train_acc, precisions, recalls, fscores, accuracies, specificities, all_accuracy, epoch):
            print_log('\nepoch %d: train_acc: %f' % (epoch, train_acc))
            print_log(
                "epoch %d: precision: %s, recall: %s, fscore: %s, accuracy: %s, specificity: %s, all_accuracy: %s\n" % (
                    epoch, str(precisions), str(recalls), str(fscores), str(accuracies), str(specificities),
                    str(all_accuracy)))
            return

        # Training loop
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
        epochs = []
        losses = []
        train_acc = []
        test_acc = []
        num_batches_per_epoch = (len(train_y) - 1) // BATCH_SIZE + 1
        for batch_x, batch_y in batches:
            loss, step = train_step(batch_x, batch_y)
            # step = tf.train.global_step(sess, global_step)
            if step % num_batches_per_epoch == 0:
                epoch = step // num_batches_per_epoch
                acc = train_accuracy()
                test_p, test_r, test_f, test_a, test_s, test_aa, labels, predictions = test_accuracy(test_x, test_y)
                write_accuracy(acc, test_p, test_r, test_f, test_a, test_s, test_aa, epoch)
                epochs.append(epoch)
                losses.append(loss)
                train_acc.append(acc)
                test_acc.append(test_aa)
                if loss < 1e-6 or acc > 0.9999:
                    break
                saver.save(sess, os.path.join(args.summary_dir, "model.ckpt"), global_step=step)

        with open(os.path.join(args.summary_dir, "LabelsAndPredictions"), "wb") as f:
            final_result = {'labels': labels, 'predictions': predictions}
            pkl.dump(final_result, f)

        def roc_curve(x, y):
            logits, _, outputs = prediction(x, y)
            logits = np.array(logits)
            prob = logits[:, 1] - logits[:, 0]
            return metrics.roc_curve(np.array(outputs), prob, pos_label=1)

        with open(os.path.join(args.summary_dir, "LossCurve.pkl"), "wb") as f:
            loss_curve = {'step': epochs, 'loss': losses, 'train_acc': train_acc, 'test_acc': test_acc}
            pkl.dump(loss_curve, f)
        fpr, tpr, thresholds = roc_curve(test_x, test_y)
        with open(os.path.join(args.summary_dir, "RocCurveData.pkl"), "wb") as f:
            roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
            pkl.dump(roc_data, f)
        return loss_curve


def logout_config(args, train_y, test_y):
    with open(os.path.join(args.summary_dir, "accuracy.txt"), "w") as f:
        print(str(args), file=f)

        labels = list(set(train_y))
        labels.sort()
        print("train samples: %d" % len(train_y), file=f)
        for label in labels:
            print("\t class %d in train set: %d samples" % (label, list(train_y).count(label)), file=f)

        labels = list(set(test_y))
        labels.sort()
        print("test samples: %d" % len(test_y), file=f)
        for label in labels:
            print("\t class %d in test set: %d samples" % (label, list(test_y).count(label)), file=f)


def only_test(test_x, test_y, kernal_initilizer):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # with tf.Session() as sess:
        tf.set_random_seed(seed)
        BATCH_SIZE = args.batch_size
        model = biLSTM(max_input_length=args.max_input_len, num_class=len(args.hidden_ratio), input_dim=args.input_dim,
                       hidden_layer_num=args.hidden_layers, bi_direction=args.bi_directional,
                       use_attention=args.use_attention, attention_size=args.attention_size, num_hidden=args.num_hidden,
                       fc_num_hidden=args.fc_num_hidden, hidden_layer_num_bi=args.hidden_layers_bi,
                       num_hidden_bi=args.num_hidden_bi, kernel_initializer=kernal_initilizer)
        # Define training procedure
        global_step = tf.Variable(0, trainable=False)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load variables from pre-trained model
        if not args.pre_trained == "none":
            pre_trained_variables = [v for v in tf.global_variables() if "Adam" not in v.name]
            saver_pre = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state(args.summary_dir)
            saver_pre.restore(sess, ckpt.model_checkpoint_path)
        batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
        outputs = []
        predictions = []
        for batch_x, batch_y in batches:
            prediction, = sess.run([model.predictions],
                                   feed_dict={model.x: batch_x, model.y: batch_y, model.keep_prob: 1.0})
            predictions.extend(prediction.tolist())
            outputs.extend(batch_y.tolist())
        accuracy = sum(np.equal(predictions, outputs)) / len(outputs)
        print("test accuracy: %f" % accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parameters for data
    parser.add_argument("--source_data_dir", type=str, default="./source", help="dataset path")
    parser.add_argument("--array_data_dir", type=str, default="./data")
    parser.add_argument("--language", type=str, default="ch", help="ch | en")
    parser.add_argument("--method", type=str, default="g729a", help="steganography method")
    parser.add_argument("--duration", type=int, default="100", help="duration of voice (ms), from 100 to 100000")
    parser.add_argument("--hidden_ratio", nargs='+', type=int, default=[0, 100],
                        help="classes to classify, hidden ratio from 0 to 100")
    parser.add_argument("--train_data_num", type=int, default=8000,
                        help="train data samples for each label, 0: unlimited")
    parser.add_argument("--test_data_num", type=int, default=2000,
                        help="test data samples for each label, 0: unlimited")

    # parameters for model
    parser.add_argument("--model", type=str, default="biLSTM")
    parser.add_argument("--hidden_layers", type=int, default=2, help="hidden LSTM layer nums")
    parser.add_argument("--input_dim", type=int, default=3, help="input feature num for each frame ")
    parser.add_argument("--num_hidden", type=int, default=50, help="hidden LSTM cell nums in each layer")
    parser.add_argument("--bi_directional", type=str, default="False", help="whether to use bi-directional LSTM")
    parser.add_argument("--use_attention", type=str, default="True", help="whether to use attention")
    parser.add_argument("--attention_size", type=int, default=20, help="attention window size")
    parser.add_argument("--hidden_layers_bi", type=int, default=2,
                        help="hidden LSTM layer nums if bi_directional is true")
    parser.add_argument("--num_hidden_bi", type=int, default=50,
                        help="hidden LSTM cell nums in each layer if bi_directional is true")
    parser.add_argument("--fc_num_hidden", type=int, default=64, help="hidden full connect ceil nums before softmax")
    parser.add_argument("--i", type=int, default=9)

    # parameters for taining
    parser.add_argument("--pre_trained", type=str, default="none", help="whether to continue training")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="epoch num")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--keep_prob", type=float, default=0.5, help="keep prob for drop out")
    parser.add_argument("--up_sample", type=int, default=0, help="up samples for each class, 0 for non-up samples")
    parser.add_argument("--max_input_len", type=int, default=None, help="max length of frame sequence")
    args = parser.parse_args()

    frame_len = int(args.duration / 10)
    args.duration = str(args.duration) + 'ms'
    args.hidden_ratio = [str(x) for x in args.hidden_ratio]
    args.bi_directional = True if args.bi_directional.lower() in ("yes", "true", "t", "1") else False
    args.use_attention = True if args.use_attention.lower() in ("yes", "true", "t", "1") else False
    if args.max_input_len is None or args.max_input_len > frame_len:
        args.max_input_len = frame_len

    print("Preprocessing dataset...")
    start_time = time.time()
    print("\tGet file list...")
    start_time1 = time.time()
    source_folders = []
    for r in args.hidden_ratio:
        source_folders.append(
            os.path.join(args.source_data_dir, '%s_%s_%s_%s_FEAT' % (args.language, args.method, r, args.duration)))
    train_data_num, test_data_num, train_file_lists, test_file_lists = available_files(source_folders,
                                                                                       args.train_data_num,
                                                                                       args.test_data_num)
    args.train_data_num = train_data_num
    args.test_data_num = test_data_num
    print("\tGet file list done.")
    print("\tTrain data samples for each: %d. Total: %d" % (train_data_num, train_data_num * len(args.hidden_ratio)))
    print("\tTest data samples for each: %d. Total: %d" % (test_data_num, test_data_num * len(args.hidden_ratio)))
    print("\tTime use: %fs" % (time.time() - start_time1))

    print("\tRead files to array...")
    start_time1 = time.time()
    dataset_dir = os.path.join(args.array_data_dir, args.language, args.method, args.duration)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    if not os.path.exists(os.path.join(dataset_dir, 'train')):
        os.makedirs(os.path.join(dataset_dir, 'train'))
    if not os.path.exists(os.path.join(dataset_dir, 'test')):
        os.makedirs(os.path.join(dataset_dir, 'test'))

    label_map = dict()
    k = 0
    for ratio in args.hidden_ratio:
        label_map[ratio] = k
        k = k + 1
    for i, ratio in enumerate(args.hidden_ratio):
        train_data = os.path.join(dataset_dir, 'train', '%s_%s_%s_%s_FEAT_%d.npa' % (
            args.language, args.method, ratio, args.duration, args.train_data_num))
        test_data = os.path.join(dataset_dir, 'test', '%s_%s_%s_%s_FEAT_%d.npa' % (
            args.language, args.method, ratio, args.duration, args.test_data_num))
        train_x.append(load_features(train_file_lists[i], train_data))
        train_y.append(np.array([label_map[ratio]] * args.train_data_num))
        test_x.append(load_features(test_file_lists[i], test_data))
        test_y.append(np.array([label_map[ratio]] * args.test_data_num))
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    train_x, train_y = shuffle(train_x, train_y, random_state=0)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    print("\tRead files to array done.")
    print("\tTime use: %fs" % (time.time() - start_time1))
    print("Dataset ready. Time use:%fs" % (time.time() - start_time))

    start_time = time.time()
    model_dir = os.path.join(args.model, args.language, args.method, args.duration)
    path = os.path.join(model_dir, '_'.join([str(x) for x in args.hidden_ratio]) + '_ratio_' + str(args.train_data_num))
    if args.bi_directional:
        path = os.path.join(path, 'bi_directional_' + str(args.hidden_layers_bi) + '_' + str(args.num_hidden_bi))
    if os.path.exists(path) is not True:
        os.makedirs(path)
    args.summary_dir = path
    args.model_dir = model_dir

    logout_config(args, train_y, test_y)
    kernal_initilizers = [tf.glorot_uniform_initializer(), tf.glorot_normal_initializer()]
    kernal_initilizer_names = ['glorot_uniform', 'glorot_normal']
    for stddev in [0.01, 0.1, 1]:
        kernal_initilizers.append(tf.random_normal_initializer(stddev=stddev))
        kernal_initilizer_names.append('random_normal_%f' % stddev)
        kernal_initilizers.append(tf.orthogonal_initializer(gain=stddev))
        kernal_initilizer_names.append('orthogonal_%f' % stddev)
        kernal_initilizers.append(tf.variance_scaling_initializer(scale=10 * stddev))
        kernal_initilizer_names.append('variance_scaling_%f' % (10 * stddev))
        kernal_initilizers.append(tf.uniform_unit_scaling_initializer(factor=stddev))
        kernal_initilizer_names.append('uniform_scaling_%f' % stddev)
        kernal_initilizers.append(tf.random_uniform_initializer(maxval=stddev))
        kernal_initilizer_names.append('random_uniform_%f' % stddev)
    # for ki, name in zip(kernal_initilizers, kernal_initilizer_names):
    # loss_curve = train(train_x, train_y, test_x, test_y, args, kernal_initilizer=kernal_initilizers[args.i])
    only_test(test_x, test_y, kernal_initilizer=kernal_initilizers[args.i])
    # with open(os.path.join('hyper', 'initializer', kernal_initilizer_names[args.i]), 'wb') as f:
    #     pkl.dump(loss_curve, file=f)

    # train(train_x, train_y, test_x, test_y, args)
    print("tf train done, time use: %fs" % (time.time() - start_time))

    # start_time = time.time()
    # from keras.models import Sequential
    # from keras.layers import Dense, Activation
    # from keras.layers import CuDNNLSTM
    #
    # print("Build model")
    # model = Sequential()
    # # model.add(LSTM(50, input_shape=(frame_len, 3), return_sequences=True))
    # # model.add(LSTM(50))
    # model.add(CuDNNLSTM(50, input_shape=(frame_len, 3), return_sequences=True))
    # model.add(CuDNNLSTM(50))
    # model.add(Dense(1, activation='sigmoid'))
    # # model.add(Activation('sigmoid'))
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    #
    # print("Train")
    # for i in range(args.num_epochs):
    #     model.fit(train_x, train_y, batch_size=args.batch_size, nb_epoch=1, shuffle=False, verbose=2,
    #               validation_data=(test_x, test_y))
    #     if i % 10 == 0:
    #         model.save('model_%d.h5' % (i + 1))
    # model.save('model_%d.h5' % (i + 1))
    # print("keras train done, time use: %fs" % (time.time() - start_time))
    #
    # start_time = time.time()
    # print("Build model")
    # model = Sequential()
    # # model.add(LSTM(50, input_shape=(frame_len, 3), return_sequences=True))
    # # model.add(LSTM(50))
    # model.add(CuDNNLSTM(50, input_shape=(frame_len, 3), return_sequences=True))
    # model.add(CuDNNLSTM(50))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    #
    # print("Train")
    # for i in range(args.num_epochs):
    #     model.fit(train_x, train_y, batch_size=args.batch_size, nb_epoch=1, shuffle=False, verbose=2,
    #               validation_data=(test_x, test_y))
    #     if i % 10 == 0:
    #         model.save('model_%d.h5' % (i + 1))
    # model.save('model_%d.h5' % (i + 1))
    # print("keras train done, time use: %fs" % (time.time() - start_time))
