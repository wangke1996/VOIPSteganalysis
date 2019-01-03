import numpy as np
import argparse
import os
import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input dataset path")
    parser.add_argument("--output_dir", type=str, help="output path for numpy array")
    args = parser.parse_args()
    print("start...")
    file_list = os.listdir(args.input_dir)
    print("%d files in total" % len(file_list))
    if len(file_list) == 0:
        print("empty folder: %s" % args.input_dir)
        raise Exception("empty data folder detected")
    file_list = [os.path.join(args.input_dir, x) for x in file_list]
    output_file = os.path.join(args.output_dir, 'all_inorder.npy')
    datas = []
    l = len(file_list)
    print("begin...")
    for i, file in enumerate(file_list):
        datas.append(np.loadtxt(file))
        if i % 50 == 49:
            print("%d/%d" % (i + 1, l))
            sys.stdout.flush()
    print("%d/%d" % (i + 1, l))
    with open(output_file, 'wb') as f:
        np.save(f, np.array(datas))
    print('done')
