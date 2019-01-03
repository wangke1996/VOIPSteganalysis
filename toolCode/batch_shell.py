import os
import multiprocessing as mp
import itertools
import argparse
import time


# shell = 'train1.py'
# thread_num = 5
# arg_names = ['num_epochs', 'i']
# arg_values = [['30'], [str(x) for x in range(17)]]


def do_task(task):
    print(os.system(task))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", type=str, default='../train.py', help='python shell name')
    parser.add_argument("--arg_names", nargs='+', type=str, help="arg names")
    parser.add_argument("--arg_values", action='append', nargs='+', type=str, help="arg values for each arg")
    parser.add_argument("--thread_num", type=int, default=-1, help="max thread nums")
    parser.add_argument("--log_folder", type=str, default='none', help='log folder, none for time stamp folder')
    args = parser.parse_args()

    shell = 'python ' + args.shell
    arg_names = ['--' + x for x in args.arg_names]
    arg_values = args.arg_values
    thread_num = args.thread_num
    if len(arg_names) != len(arg_values):
        print('wrong parameters! %d args, but %d lists of arg values!' % (len(arg_names), len(arg_values)))

    tasks = []
    task_id = 0
    if args.log_folder == 'none':
        time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.strptime(time.ctime()))
        log_folder = os.path.join('logs', time_stamp)
    else:
        log_folder = os.path.join('logs', args.log_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    for arg_value_pair in itertools.product(*arg_values):
        task = shell
        for arg_name, arg_value in zip(arg_names, arg_value_pair):
            task = task + ' %s=%s' % (arg_name, arg_value)
        log_file = os.path.join(log_folder, str(task_id))
        print("task: %s\t log_file: %s" % (task, log_file))
        task = task + ' > %s 2>&1' % log_file
        tasks.append(task)
        task_id = task_id + 1
        print(task)
    if thread_num == -1:
        thread_num = len(tasks)
    pool = mp.Pool(processes=thread_num)
    for task in tasks:
        pool.apply_async(do_task, (task,))
    pool.close()
    pool.join()
