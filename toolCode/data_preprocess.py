import os
import multiprocessing as mp

shell = './samples2nparray.py'
source_folder = '/data/wangke/VOIP/data'
npa_folder = '/data/wangke/VOIP/npa'
languages = ['ch', 'en']
methods = ['g729a']
ratios = [str(x) for x in list(range(0, 101, 10))]
durations = ["%dms" % x for x in (list(range(100, 1001, 100)) + list(range(2000, 10001, 1000)))]


def do_task(task):
    print(os.system(task))


if __name__ == '__main__':
    tasks = []
    for l in languages:
        for m in methods:
            for r in ratios:
                for d in durations:
                    sub_folder = '%s_%s_%s_%s_FEAT' % (l, m, r, d)
                    input_folder = os.path.join(source_folder, sub_folder)
                    if not os.path.exists(input_folder):
                        print('input folder does not exist: %s' % sub_folder)
                        continue
                    output_folder = os.path.join(npa_folder, sub_folder)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    tasks.append('python %s --input_dir=%s --output_dir=%s > %s.log' % (
                        shell, input_folder, output_folder, os.path.join(output_folder, 'preprocess')))
    print("%d tasks in toatal" % len(tasks))
    pool = mp.Pool(processes=10)
    for task in tasks:
        pool.apply_async(do_task, (task,))
    pool.close()
    pool.join()
