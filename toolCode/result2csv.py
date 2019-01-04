import os
import time
import argparse
import pickle as pkl
import numpy as np
import itertools
from sklearn import metrics


def read_result(folder):
    result_list = []
    direction_folders = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
    for direction in direction_folders:
        direction_dict = {direction: []}
        pre_path = os.path.join(folder, direction)
        attention_folders = [x for x in os.listdir(pre_path) if os.path.isdir(os.path.join(pre_path, x))]
        for attention in attention_folders:
            with open('LabelsAndPredictions', 'rb') as f:
                labels_and_predictions = pkl.load(f)
            labels = np.array(labels_and_predictions['labels'])
            predictions = np.array(labels_and_predictions['predictions'])
            tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels, y_pred=predictions).ravel()
            acc = (tp + tn) / (tp + fp + fn + tp)
            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)
            result = {'acc': acc, 'fpr': fpr, 'fnr': fnr}
            direction_dict[direction].append({attention: result})
        result_list.append(direction_dict)
    return result_list


def get_dict(result_dict, arguments):
    d = result_dict
    for arg in arguments:
        if arg not in d:
            d1 = dict()
            d[arg] = d1
        else:
            d1 = d[arg]
        d = d1
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parameters for data
    parser.add_argument("--result_dir", type=str, default="/data1/wangke/VOIP/results")
    parser.add_argument("--models", nargs='+', type=str, default=['biLSTM'])
    parser.add_argument("--languages", nargs='+', type=str, default=["ch"], help="language list (ch en)")
    parser.add_argument("--methods", nargs='+', type=str, default=["g729a"], help="steganography methods")
    parser.add_argument("--durations", nargs='+', type=int, default=[500],
                        help="duration of voice (ms), from 100 to 100000")
    parser.add_argument("--hidden_ratios", nargs='+', type=int, default=[20, 40, 60, 80, 100],
                        help="classes to classify, hidden ratio from 0 to 100")
    parser.add_argument('--directions', nargs='+', type=str, default=['uni', 'bi'])
    parser.add_argument('--layers_cells', nargs='+', type=str, default=['2_50'], help="lstm layer and cell nums")
    parser.add_argument('--attentions', nargs='+', type=int, default=[0, 20],
                        help="attention window size, 0 for no attention")
    args = parser.parse_args()

    results = dict()
    parameters = itertools.product(args.models, args.languages, args.methods, args.durations, args.hidden_ratios,
                                   args.directions, args.layers_cells, args.attentions)
    for model, language, method, duration, hidden_ratio, direction, layer_cell, attention in parameters:
        result_dict = get_dict(results,
                               [model, language, method, duration, hidden_ratio, direction, layer_cell, attention])
        file_path = os.path.join(args.result_dir, model, language, method, "%dms" % duration,
                                 "0_%d_ratio_-1" % hidden_ratio, "%s_directional_%s" % (direction, layer_cell),
                                 'no_attention' if attention == 0 else 'attention_%d' % attention,
                                 'LabelsAndPredictions')
        if not os.path.exists(file_path):
            print("Error, result file not exists: %s" % file_path)
            continue
        with open(file_path, 'rb') as f:
            labels_and_predictions = pkl.load(f)
        labels = labels_and_predictions['labels']
        predictions = labels_and_predictions['predictions']
        tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels, y_pred=predictions).ravel()
        acc = (tp + tn) / (tp + fp + fn + tp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        result_dict['acc'] = acc
        result_dict['fpr'] = fpr
        result_dict['fnr'] = fnr
    for hidden_ratio in args.hidden_ratios:
        time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.strptime(time.ctime()))
        output_dir = os.path.join('results', 'fix_hidden_ratio', str(hidden_ratio))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, time_stamp + '.csv')
        f = open(output_file, 'w', encoding='utf8')
        f.write('HIDDEN_RATIO,MODEL,LANGUAGE,METHOD,DIRECTION,LAYER_CELL,ATTENTION,METRIC')
        for duration in args.durations:
            f.write(',%dms' % duration)
        f.write('\n')
        parameters = itertools.product(args.models, args.languages, args.methods, args.directions, args.layers_cells,
                                       args.attentions, ['acc', 'fpr', 'fnr'])
        for model, language, method, direction, layer_cell, attention, metric in parameters:
            f.write('%d,%s,%s,%s,%s,%s,%s,%s' % (
                hidden_ratio, model, language, method, direction, layer_cell, 'no' if attention == 0 else attention,
                metric))
            for duration in args.durations:
                result = get_dict(results,
                                  [model, language, method, duration, hidden_ratio, direction, layer_cell, attention,
                                   metric])
                try:
                    len(result)
                    f.write(',')  # expected a value, got dict, indicate that no value available
                except Exception as e:
                    f.write(',%f' % result)
            f.write('\n')
        f.close()

    for duration in args.durations:
        time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.strptime(time.ctime()))
        output_dir = os.path.join('results', 'fix_duration', str(duration) + 'ms')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, time_stamp + '.csv')
        f = open(output_file, 'w', encoding='utf8')
        f.write('DURATION,MODEL,LANGUAGE,METHOD,DIRECTION,LAYER_CELL,ATTENTION,METRIC')
        for hidden_ratio in args.hidden_ratios:
            f.write(',%d' % hidden_ratio)
        f.write('\n')
        parameters = itertools.product(args.models, args.languages, args.methods, args.directions, args.layers_cells,
                                       args.attentions, ['acc', 'fpr', 'fnr'])
        for model, language, method, direction, layer_cell, attention, metric in parameters:
            f.write('%dms,%s,%s,%s,%s,%s,%s,%s' % (
                duration, model, language, method, direction, layer_cell, 'no' if attention == 0 else attention,
                metric))
            for hidden_ratio in args.hidden_ratios:
                result = get_dict(results,
                                  [model, language, method, duration, hidden_ratio, direction, layer_cell, attention,
                                   metric])
                try:
                    len(result)
                    f.write(',')  # expected a value, got dict, indicate that no value available
                except Exception as e:
                    f.write(',%f' % result)
            f.write('\n')
        f.close()
