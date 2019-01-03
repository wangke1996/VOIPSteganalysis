import tensorflow as tf
import argparse
import os
import pickle as pkl
import time
import numpy as np
from model.biRNN import biLSTM
from data_helper import available_files, load_features, batch_iter
from sklearn import metrics
from sklearn.utils import shuffle
import sys

seed = 2333
np.random.seed(seed)
tf.set_random_seed(seed)


# os.environ['CUDA_VISIBLE_DEVICES']='1'
def train(train_x, train_y, test_x, test_y, args):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # with tf.Session() as sess:
        tf.set_random_seed(seed)
        BATCH_SIZE = args.batch_size
        NUM_EPOCHS = args.num_epochs
        model = biLSTM(max_input_length=args.max_input_len, num_class=len(args.hidden_ratio), input_dim=args.input_dim,
                       hidden_layer_num=args.hidden_layers, bi_direction=args.bi_directional,
                       num_hidden=args.num_hidden, fc_num_hidden=args.fc_num_hidden, dropout=1.0 - args.keep_prob,
                       hidden_layer_num_bi=args.hidden_layers_bi, num_hidden_bi=args.num_hidden_bi,
                       use_attention=args.use_attention, attention_size=args.attention_size)
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
            sys.stdout.flush()
            return

        def train_step(batch_x, batch_y):
            feed_dict = {
                model.x: batch_x,
                model.y: batch_y,
                model.keep_prob: args.keep_prob  # 0.5
            }

            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)
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
            elif step % 500 == 0:
                print_log("step %d: loss = %f" % (step, loss))

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
        print(str(args))

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
                       num_hidden_bi=args.num_hidden_bi)
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
    parser.add_argument("--array_data_dir", type=str, default="/data/wangke/VOIP/npa")
    parser.add_argument("--language", type=str, default="ch", help="ch | en")
    parser.add_argument("--method", type=str, default="g729a", help="steganography method")
    parser.add_argument("--duration", type=int, default="100", help="duration of voice (ms), from 100 to 100000")
    parser.add_argument("--hidden_ratio", nargs='+', type=int, default=[0, 10],
                        help="classes to classify, hidden ratio from 0 to 100")
    parser.add_argument("--train_data_num", type=int, default=-1,
                        help="train data samples for each label, -1: unlimited")
    parser.add_argument("--test_data_num", type=int, default=-1, help="test data samples for each label, -1: unlimited")

    # parameters for model
    parser.add_argument("--result_dir", type=str, default="/data/wangke/VOIP/results")
    parser.add_argument("--model", type=str, default="biLSTM")
    parser.add_argument("--hidden_layers", type=int, default=2, help="hidden LSTM layer nums")
    parser.add_argument("--input_dim", type=int, default=3, help="input feature num for each frame ")
    parser.add_argument("--num_hidden", type=int, default=50, help="hidden LSTM cell nums in each layer")
    parser.add_argument("--bi_directional", type=str, default="True", help="whether to use bi-directional LSTM")
    parser.add_argument("--hidden_layers_bi", type=int, default=2,
                        help="hidden LSTM layer nums if bi_directional is true")
    parser.add_argument("--num_hidden_bi", type=int, default=50,
                        help="hidden LSTM cell nums in each layer if bi_directional is true")
    parser.add_argument("--use_attention", type=str, default="True", help="whether to use attention")
    parser.add_argument("--attention_size", type=int, default=20, help="attention window size")
    parser.add_argument("--fc_num_hidden", type=int, default=64, help="hidden full connect ceil nums before softmax")

    # parameters for taining
    parser.add_argument("--gpu_id", type=str, default=4, help="cuda device number")
    parser.add_argument("--pre_trained", type=str, default="none", help="whether to continue training")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="epoch num")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--keep_prob", type=float, default=0.5, help="keep prob for drop out")
    parser.add_argument("--max_input_len", type=int, default=None, help="max length of frame sequence")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    frame_len = args.duration / 10
    args.duration = str(args.duration) + 'ms'
    args.hidden_ratio = [str(x) for x in args.hidden_ratio]
    if len(args.hidden_ratio) == 1:
        args.hidden_ratio = ['0'] + args.hidden_ratio
    args.bi_directional = True if args.bi_directional.lower() in ("yes", "true", "t", "1") else False
    args.use_attention = True if args.use_attention.lower() in ("yes", "true", "t", "1") else False
    if args.max_input_len is None or args.max_input_len > frame_len:
        args.max_input_len = frame_len

    print("Preprocessing dataset...")
    start_time = time.time()
    print("\tLoad arrays...")
    start_time1 = time.time()


    def data_from_folder(folder):
        print("\t\tload data from folder %s..." % folder)
        time_begin = time.time()
        train_folder = os.path.join(folder, 'train')
        test_folder = os.path.join(folder, 'test')
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        train_file = os.path.join(train_folder, 'all_train.npy')
        test_file = os.path.join(test_folder, 'all_test.npy')
        if not os.path.exists(train_file):
            print("\t\tcreate shuffled samples for the first time")
            all_samples = np.load(os.path.join(folder, 'all_inorder.npy'))
            np.random.shuffle(all_samples)
            train_data = all_samples[:round(0.8 * len(all_samples))]
            np.save(train_file, train_data)
            test_data = all_samples[-round(0.2 * len(all_samples)):]
            np.save(test_file, test_data)
        else:
            train_data = np.load(train_file)
            test_data = np.load(test_file)
        train_data = train_data[:args.train_data_num]
        test_data = test_data[:args.test_data_num]
        print("\t\tdata loaded from folder, time use: %fs" % (time.time() - time_begin))
        return train_data, test_data


    train_x = []
    test_x = []
    for r in args.hidden_ratio:
        folder = os.path.join(args.array_data_dir, '%s_%s_%s_%s_FEAT' % (args.language, args.method, r, args.duration))
        train_data, test_data = data_from_folder(folder)
        train_x.append(train_data)
        test_x.append(test_data)
    print("\tLoad arrays done, time use: %fs" % (time.time() - start_time))
    available_train_samples = min([len(x) for x in train_x])
    available_test_samples = min([len(x) for x in test_x])
    train_x = [x[:available_train_samples] for x in train_x]
    test_x = [x[:available_test_samples] for x in test_x]

    label_map = dict()
    k = 0
    for ratio in args.hidden_ratio:
        label_map[ratio] = k
        k = k + 1
    train_y = [[label_map[r]] * available_train_samples for r in args.hidden_ratio]
    test_y = [[label_map[r]] * available_test_samples for r in args.hidden_ratio]

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    train_x, train_y = shuffle(train_x, train_y, random_state=0)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)

    print("\tTrain data samples for each: %d. Total: %d" % (
        available_train_samples, available_train_samples * len(args.hidden_ratio)))
    print("\tTest data samples for each: %d. Total: %d" % (
        available_test_samples, available_test_samples * len(args.hidden_ratio)))
    print("Dataset ready. Time use:%fs" % (time.time() - start_time))

    model_dir = os.path.join(args.result_dir, args.model, args.language, args.method, args.duration)
    path = os.path.join(model_dir, '_'.join([str(x) for x in args.hidden_ratio]) + '_ratio_' + str(args.train_data_num))
    if args.bi_directional:
        path = os.path.join(path, 'bi_directional_' + str(args.hidden_layers_bi) + '_' + str(args.num_hidden_bi))
    else:
        path = os.path.join(path, 'uni_directional_' + str(args.hidden_layers) + '_' + str(args.num_hidden))
    if args.use_attention:
        path = os.path.join(path, 'attention_' + str(args.attention_size))
    else:
        path = os.path.join(path, 'no_attention')
    if not os.path.exists(path):
        os.makedirs(path)
    args.summary_dir = path
    args.model_dir = model_dir

    logout_config(args, train_y, test_y)
    if args.model == 'biLSTM':
        train(train_x, train_y, test_x, test_y, args)
    else:
        start_time = time.time()
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        from keras.layers import CuDNNLSTM

        start_time = time.time()
        print("Build model")
        model = Sequential()
        # model.add(LSTM(50, input_shape=(frame_len, 3), return_sequences=True))
        # model.add(LSTM(50))
        model.add(CuDNNLSTM(50, input_shape=(frame_len, 3), return_sequences=True))
        model.add(CuDNNLSTM(50))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

        print("Train")
        for i in range(args.num_epochs):
            model.fit(train_x, train_y, batch_size=args.batch_size, nb_epoch=1, shuffle=False, verbose=1,
                      validation_data=(test_x, test_y))
            if i % 1 == 0:
                model.save('model_%d.h5' % (i + 1))
        print("keras train done, time use: %fs" % (time.time() - start_time))
