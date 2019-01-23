# -*- coding: utf-8 -*-
# python3
# calculate FID asynchronously
# search all the *.npz file under ROOT_EXP_DIR to calculate FID
# and save results in one *.txt file
from __future__ import absolute_import, division, print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import gzip, pickle
import tensorflow as tf
from scipy.misc import imread
from scipy import linalg
import pathlib
#import urllib
import time
from tqdm import trange
import argparse
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('--hashmin',type=int,default=0,help='used to adjust workload.if hashmin <= hash(file) % 100 < hashmax, file will be proccessed by this instance')
parser.add_argument('--hashmax',type=int,default=100,help='used to adjust workload.if hashmin <= hash(file) % 100 < hashmax, file will be proccessed by this instance')
parser.add_argument('--shuffle_seed',type=int,default=123456,help='shuffle')
parser.add_argument('--root_dir',type=str,default='',help='the dir which contains all exp folders')
parser.add_argument('--iter',type=str,default='',help='only compute specific iter')
args = parser.parse_args()

assert(args.hashmin < args.hashmax)

ROOT_EXP_DIR = args.root_dir or 'exp_output7/'
SAVE_LOG_FILE = ROOT_EXP_DIR+'/fid_log.txt'
LOAD_LOG_FILE = ROOT_EXP_DIR+'/fid_log.txt'
SHUFFLE_SEED = args.shuffle_seed
VERBOSE = True
WRITE_BUFFER_SIZE = 2 #can be 1 to any positive number. 
error_count = 0
_result_path = None

if ROOT_EXP_DIR[-1] != '/':
    ROOT_EXP_DIR += '/'

class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')
#-------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py

def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if tf.__version__ == '1.2.1':
                if shape._dims is not None:
                  shape = [s.value for s in shape]
                  new_shape = []
                  for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                      new_shape.append(None)
                    else:
                      new_shape.append(s)
                  o._shape = tf.TensorShape(new_shape)
            elif tf.__version__ == '1.12.0':
                if shape._dims != []:
                  shape = [s.value for s in shape]
                  new_shape = []
                  for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                      new_shape.append(None)
                    else:
                      new_shape.append(s)
                  o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
            else:
                raise NotImplementedError(
                        "tf version not tested, try 1.12.0 or 1.2.1")
    return pool3
#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=8, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    for i in trange(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
    if verbose:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
#    print('diff',diff)
    # product might be almost singular
    r=linalg.sqrtm(np.random.rand(2048,2048))
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        # warnings.warn(msg)
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------



def calculate_activation_statistics(images, sess, batch_size=8, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
#-------------------------------------------------------------------------------

def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        # from urllib import request
        import urllib as request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def _handle_path(path, sess,batch_size=32,precalc=True):
    global error_count
    if path.endswith('.npz') and precalc:
        while True:
            try:
                f = np.load(path)
                m, s = f['mu'][:], f['sigma'][:]
                f.close()
                break
            except OSError:
                print("OSError, retry after 60s")
                error_count += 1
                time.sleep(60)
            
    # load GAN saved output image array.
    elif path.endswith('.npz') and not precalc:
        print("Loading " + path)
        while True:
            try:
                f = np.load(path)
                x = f['arr_0']
                f.close
                break
            except OSError:
                print("OSError, retry after 60s")
                error_count += 1
                time.sleep(60)
        assert(x.shape[0] > 0)
        m, s = calculate_activation_statistics(x, sess,batch_size=batch_size)
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        assert(len(files) > 0)
        x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
        m, s = calculate_activation_statistics(x, sess,batch_size=batch_size)
    return m, s

def get_npz_path_list(root_exp_dir='exp_output/',iter_num=''):
    iter_pattern = '_iter' + iter_num + '.npz' if iter_num!='' else ''
    global error_count
    while True:
        try:
            npz_path_list = []
            exp_dir_list = filter(os.path.isdir,\
                        [root_exp_dir+p for p in os.listdir(root_exp_dir)])
            for exp_dir in exp_dir_list:
                exp_files = os.listdir(exp_dir)
                npz_path_list += [exp_dir+'/' + f for f in exp_files\
                                 if f.endswith('.npz') and iter_pattern in f]
            break
        except OSError:
            print("OSError, retry after 30s")
            error_count += 1
            time.sleep(30)
    return npz_path_list
    

def get_log():
    score_dict={}
    try:
        with open(LOAD_LOG_FILE)  as f:
            fstr = f.readlines()
        for line in fstr:
            if line[0] != '#':
                npz,fid = line.split()
                score_dict[npz] = float(fid)
        print('score_dict loaded from {}'.format(LOAD_LOG_FILE))
    except:
        print('INFO: No log found')
    return score_dict

def my_cmp(x,y):
    cmp = lambda a,b:((a > b) - (a < b))
    x1 = x[:x.rfind('/')]
    y1 = y[:y.rfind('/')]
    return cmp(x1,y1) or cmp(int(x[x.rfind('iter')+4:x.rfind('.npz')]),\
                       int(y[y.rfind('iter')+4:y.rfind('.npz')]))  
import functools
my_key = functools.cmp_to_key(my_cmp)
#score_dict = {}
score_dict = get_log()

def get_hash(npz_file,seed=0):
    seed1 = int(hashlib.md5(npz_file.encode()).hexdigest(),base=16) % 2**30
    np.random.seed(seed+seed1)
    return np.random.randint(0,100)

inception_path = check_or_download_inception('imagenet_inception/')
create_inception_graph(str(inception_path))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# limit gpu memory usage. Performance increases little even using more mem.
config.gpu_options.per_process_gpu_memory_fraction = 0.2 #SHF
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        time.sleep(np.random.randint(5,15)) #workaround to reduce disk load when many instances runing simultaneously
        npz_path_list = get_npz_path_list(root_exp_dir=ROOT_EXP_DIR,iter_num=args.iter)
        n_fth = len(npz_path_list) - len(score_dict)
        if n_fth > 0:
            print('# files to handle = {}'.format(n_fth))
        changed = 0
        hash_list = []
        for npz_file in npz_path_list:
            int_hash = get_hash(npz_file,SHUFFLE_SEED)
            hash_list.append(int_hash)
        hist, _ = np.histogram(hash_list,bins=5,range=[0,100])
        print("hash distrib.:\n{}".format(hist))
        for npz_file in npz_path_list:
            
            if not (npz_file in score_dict):
                pre_calc_path = None
                if 'cifar' in npz_file:
                    pre_calc_path = 'pre_calc/fid_stats_cifar10_train.npz'
                elif 'mnist32' in npz_file:
                    pre_calc_path = 'pre_calc/fid_stats_mnist32.npz'
                elif 'fashion32' in npz_file:
                    pre_calc_path = 'pre_calc/fid_stats_fashion32.npz'
                elif '-celeba-' in npz_file:
                    pre_calc_path = 'pre_calc/fid_stats_1.0celeba.npz'
                elif '-cartoon-' in npz_file:
                    pre_calc_path = 'pre_calc/fid_stats_1.0cartoon.npz'
                elif '-0.7celeba-0.3cartoon-' in npz_file:
                    pre_calc_path = 'pre_calc/fid_stats_0.7celeba-0.3cartoon.npz'
                elif '-0.8celeba-0.2cartoon-' in npz_file:
                    pre_calc_path = 'pre_calc/fid_stats_0.8celeba-0.2cartoon.npz'
                int_hash = get_hash(npz_file,SHUFFLE_SEED)
                if args.hashmin <= int_hash and int_hash < args.hashmax:
                    if pre_calc_path:
                        # wait until the npz file has been created over 3 min
                        while time.time() - os.path.getctime(npz_file) < 180:
                            time.sleep(10)
                        m1,s1 = _handle_path(pre_calc_path,sess,precalc=True)
                        m2,s2 = _handle_path(npz_file,sess,precalc=False,batch_size=32)
                        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
                        print('>>>\t{}\t{}'.format(npz_file,fid_value))
                        score_dict[npz_file] = fid_value
                        changed += 1
                        if changed >= WRITE_BUFFER_SIZE:
                            break;
                    else:
                        if VERBOSE:
                            print('ignore {} because of no precalc stats.'.format(npz_file))
                else:
                    if VERBOSE:
                        if pre_calc_path:
                            print(('{} is omited by instance[{},{})'+\
                             ' bacause its hash is {}')\
                             .format(npz_file,args.hashmin,args.hashmax,int_hash))
                        else:
                            print('ignore {} because of no precalc stats.'.format(npz_file))
        if changed:
            print('\n'*4)
            records = []
            #update score dict
            score_dict_now = get_log()
            for k in score_dict_now:
                score_dict[k] = score_dict_now[k]
            for npz_file in score_dict:
                fid_value = score_dict[npz_file]
                records.append('{}\t{}\n'.format(npz_file,fid_value))
            records.sort(key=my_key)
            while True:
                try:
                    with open(SAVE_LOG_FILE,'w')  as f:
                        f.write('# image_npz    fid\n')
                        f.writelines(records)
                        print('log saved')
                        break
                except OSError:
                    print("OSError when saving log, retry after 60s")
                    error_count += 1
                    time.sleep(60)
                except IOError:
                    print("IOError when saving log, retry after 60s")
                    error_count += 1
                    time.sleep(60)
            for rec in records:
                print('  '+rec[:-1])
        print('#### Idling ####; ErrorCount = {}'.format(error_count))
                
