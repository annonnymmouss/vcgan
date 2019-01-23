import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import collections
import time
import cPickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
    _iter[0] += 1

def plot(name, value):
    _since_last_flush[name][_iter[0]] = value

def load(log_dir='',iter_start=0):
    if os.path.exists(log_dir+'/'+'log.pkl'):
        with open(log_dir+'/'+'log.pkl', 'rb') as f:
            last_log = pickle.load(f)
        for k,v in last_log.items():
            _since_beginning[k] = v
        _iter[0] = iter_start
        print(log_dir+'/'+'log.pkl loaded')


def flush(log_dir='',header=None):
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{} = {:.4f}".format(name, np.mean(vals.values())))
        _since_beginning[name].update(vals)

        x_vals = np.sort(_since_beginning[name].keys())
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.grid()
        while True:
            try:
                plt.savefig(log_dir+'/'+name.replace(' ', '_')+'.png')
                break
            except OSError:
                print("OSError when saving log, retry after 60s")
                time.sleep(60)
            except IOError:
                print("IOError when saving log, retry after 60s")
                time.sleep(60)

    if header:
        print(header)
    print "iter_{:06d} {}".format(_iter[0], "\t".join(prints))
    _since_last_flush.clear()
    while True:
        try:
            with open(log_dir+'/'+'log.pkl', 'wb') as f:
                pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
            break
        except OSError:
            print("OSError when saving log, retry after 60s")
            time.sleep(60)
        except IOError:
            print("IOError when saving log, retry after 60s")
            time.sleep(60)