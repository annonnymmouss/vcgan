# -*- coding: utf-8 -*-
# python2
# calculate Inception Score asynchronously
# search all the *.npz file under ROOT_EXP_DIR to calculate Inception Score
# and save results in one *.txt file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tflib.inception_score as i_score
import numpy as np
import argparse
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('--hashmin',type=int,default=0,help='used to adjust workload.if hashmin <= hash(file) % 100 < hashmax, file will be proccessed by this instance')
parser.add_argument('--hashmax',type=int,default=100,help='used to adjust workload.if hashmin <= hash(file) % 100 < hashmax, file will be proccessed by this instance')
parser.add_argument('--shuffleseed',type=int,default=123456,help='shuffle')
parser.add_argument('--root_dir',type=str,default='',help='the dir which contains all exp folders')
args = parser.parse_args()

assert(args.hashmin < args.hashmax)

ROOT_EXP_DIR = args.root_dir or 'exp_output7/'
SAVE_LOG_FILE = ROOT_EXP_DIR+'/inception_score_log.txt'
LOAD_LOG_FILE = ROOT_EXP_DIR+'/inception_score_log.txt'
SHUFFLE_SEED = args.shuffleseed
WRITE_BUFFER_SIZE = 2 #can be 1 to any positive number. 
def get_log():
    score_dict={}
    try:
        with open(LOAD_LOG_FILE)  as f:
            fstr = f.readlines()
        for line in fstr:
            if line[0] != '#':
                npz,score,std = line.split()
                score_dict[npz] = (float(score),float(std))
        print('score_dict loaded from {}'.format(LOAD_LOG_FILE))
    except:
        print('INFO: No log found')
    return score_dict

def get_inception_score_from_npz(image_npz_path):
    print('Loading ' + image_npz_path)
    all_samples = np.load(image_npz_path)['arr_0']
    #return (6.66666666,1.11111111)
    return i_score.get_inception_score(list(all_samples),splits=10,bs=32)

def get_npz_path_list(root_exp_dir=ROOT_EXP_DIR):
    npz_path_list = []
    exp_dir_list = filter(os.path.isdir,\
                          [root_exp_dir+p for p in os.listdir(root_exp_dir)])
    for exp_dir in exp_dir_list:
        exp_files = os.listdir(exp_dir)
#        npz_path_list += [os.path.realpath(exp_dir+'/' + f) \
#                          for f in exp_files if f.endswith('.npz')]
        npz_path_list += [exp_dir+'/' + f \
                          for f in exp_files if f.endswith('.npz')]
    return npz_path_list

def my_cmp(x,y):
    x1 = x[:x.rfind('/')]
    y1 = y[:y.rfind('/')]
    return cmp(x1,y1) or cmp(int(x[x.rfind('iter')+4:x.rfind('.npz')]),\
                       int(y[y.rfind('iter')+4:y.rfind('.npz')]))  

#score_dict = {}
score_dict = get_log()

def get_hash(npz_file,seed=0):
    seed1 = int(hashlib.md5(npz_file.encode()).hexdigest(),base=16) % 2**30
    np.random.seed(seed+seed1)
    return np.random.randint(0,100)

while True:
    time.sleep(5 + np.random.randint(0,11)) #workaround to reduce disk load when many instances runing simultaneously
    npz_path_list = get_npz_path_list(root_exp_dir=ROOT_EXP_DIR)
    n_fth = len(npz_path_list) - len(score_dict)
    if n_fth > 0:
        print('# files to handle = {}'.format(n_fth))
    changed = 0
    for npz_file in npz_path_list:
        if not 'cifar' in npz_file:
            print('ignore {}'.format(npz_file))
            continue
        
        if not score_dict.has_key(npz_file):
            int_hash = get_hash(npz_file,SHUFFLE_SEED)
            if args.hashmin <= int_hash and int_hash < args.hashmax:
                # wait until the npz file has been created over 3 min
                while time.time() - os.path.getctime(npz_file) < 180:
                    time.sleep(10)
                score_dict[npz_file] = get_inception_score_from_npz(npz_file)
                score_,std_ = score_dict[npz_file]
                print('>>>{}\t{}\t{}'.format(npz_file,score_,std_))
                changed += 1
                if changed >= WRITE_BUFFER_SIZE:
                    break;
            else:
                print(('{} is omited by instance[{},{})'+\
                     ' bacause its hash is {}')\
                     .format(npz_file,args.hashmin,args.hashmax,int_hash))
    if changed:
        print('\n'*4)
        records = []
        #update score dict
        score_dict_now = get_log()
        for k in score_dict_now:
            score_dict[k] = score_dict_now[k]
        for npz_file in score_dict:
            score_,std_ = score_dict[npz_file]
            records.append('{}\t{}\t{}\n'.format(npz_file,score_,std_))
        records.sort(my_cmp)
        while True:
            try:
                with open(SAVE_LOG_FILE,'w')  as f:
                    f.write('# image_npz    inception_score    std\n')
                    f.writelines(records)
                print('log saved')
                break
            except OSError:
                print("OSError when saving log, retry after 60s")
                time.sleep(60)
            except IOError:
                print("IOError when saving log, retry after 60s")
                time.sleep(60)
        for rec in records:
            print('  '+rec[:-1])
    print('#### idling ####')
                
