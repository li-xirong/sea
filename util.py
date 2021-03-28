import os
import sys
import time
import logging
from functools import wraps
from collections import OrderedDict

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)


def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        if overwrite:
            logging.info('%s exists. overwrite', filename)
            return 0
        else:
            logging.info('%s exists. quit', filename)
            return 1
    return 0


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def makedirsforfile(filename):
    makedirs(os.path.dirname(filename))


def timer(fn):
    @wraps(fn)
    def compute_time(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time
        print (fn.__name__ + ' execution time: %.3f seconds\n' % elapsed_time)

        return ret
    return compute_time

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=1):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, step=step)


def perf_txt_to_excel(pattern_file_path, perf_file_dir):

    perf_file_path = os.path.join(perf_file_dir, 'perf.txt')
    excel_file_path = os.path.join(perf_file_dir, 'perf.xlsx')
    with open(pattern_file_path, 'r') as fr:
        pattern = fr.read()
    with open(perf_file_path, 'r') as fr:
        text = fr.read()
    from scanf import scanf
    pef_num_list = scanf(pattern, text)
    pef_num_list = list(map(str, pef_num_list))

    from openpyxl import Workbook
    workbook = Workbook()
    booksheet = workbook.active     
    titles = [['Text to video', '', '', '', '', '', '', 'Video to text', '', '', '', '', '', ''],['r1', 'r5', 'r10', 'medr', 'meanr', 'mir', 'mAP', 'r1', 'r5', 'r10', 'medr', 'meanr', 'mir', 'mAP']]
    for line, title_line in enumerate(titles):
        booksheet.append(title_line) 
    booksheet.append(pef_num_list)
    workbook.save(filename=excel_file_path)

    from common import logger
    logger.info('perf.txt has transformed into excel file: %s' % (excel_file_path))

if __name__ == '__main__':
    pass