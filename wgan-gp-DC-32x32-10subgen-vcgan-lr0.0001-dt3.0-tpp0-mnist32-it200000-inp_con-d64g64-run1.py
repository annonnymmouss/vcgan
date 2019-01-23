# -*- coding: utf-8 -*-
# python2.7
# resolution 32x32x3


import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.dataloader_RAM as dataloader
import tflib.plot
from tqdm import trange
import shutil
import functools
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--workmode',type=str,help='mode? train or generate',
                    default='train',choices=['train','generate'])
args = parser.parse_args()
WORKMODE = args.workmode

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
home_dir = os.path.expanduser('~')
# copy datasets to ~/datasets or make soft links to datasets in other places.
DATASETS =  [#name, ratio, path
            ['cifar',   0.0,home_dir + '/datasets/cifar-10-batches-py/'],
            ['mnist32', 1.0,home_dir + '/datasets/mnist32/'],
            ['fashion32', 0.0,home_dir + '/datasets/fashion_mnist32/'],
            ]
EXP_ROOT_DIR = 'exp_output7'
COMMENT = 'run1'
MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
ARCH = 'DC' #valid options are 'DC' or 'Res'
DDIM = 64
GDIM = 64
Z1DIM = 128
LAMBDA = 5 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
Z2DIM = N_SUBGEN = 10#  How many sub-generators in the Generator
# GEN_TYPE can be 'multigen' 'deligan' 'vcgan'
GEN_TYPE = 'vcgan'
INPLACE_COND = True
DELTA = 3.0

if GEN_TYPE == 'vcgan' and INPLACE_COND:
    Z1DIM -= N_SUBGEN
DELIGAN_INIT_SCALE = 1.0
ITERS = 200000 # How many generator iterations to train for
N_C,N_H,N_W = 3,32,32
OUTPUT_DIM = int(N_C * N_H * N_W) # Number of pixels
RELU_GP = False
TRAIN_P_START = 2000
TRAIN_P_PERIOD = 0  # train p every TRAIN_P_PERIOD iters. 0 to disable training p
LR = 1e-4
LRLD = False #enable or disable linear decay of Gen and Dis learning rate
PLR = 0.01
PLRDR = 0.98 #PLR decay rate

CALC_INCEPTION_SCORE = False
SAVE_CHECKPOINT_PERIOD = 10000
SAVE_SAMPLES_PERIOD = 10000
PREVIEW_PERIOD = 250


def datasets_to_str(ds):
    ds_str = ''
    for name,p,path in ds:
        if p != 0.0 and p < 1.0:
            ds_str += '-{}{}'.format(p,name)
        if p != 0.0 and p == 1.0:
            ds_str += '-{}'.format(name)
    return ds_str


EXPERIMENT_NAME = MODE + '-' + ARCH + '-32x32-{}subgen'.format(N_SUBGEN) + \
    '-{}'.format(GEN_TYPE) + '-lr{}'.format(LR) + ('d' if LRLD else '') +\
    ('-rgp' if RELU_GP else '') + \
    ('-dlis{}'.format(DELIGAN_INIT_SCALE) if GEN_TYPE == 'deligan' else '') + \
    ('-dt{}'.format(DELTA) if GEN_TYPE == 'vcgan' else '') + \
    '-tpp{}'.format(TRAIN_P_PERIOD) + \
    ('-plr{}-dr{}-tps{}'.format(PLR,PLRDR,TRAIN_P_START)\
     if TRAIN_P_PERIOD > 0 else '') + \
    datasets_to_str(DATASETS) + '-it{}'.format(ITERS) + \
    ('-inp_con' if GEN_TYPE == 'vcgan' and INPLACE_COND else '') +\
    '-d{}g{}-'.format(DDIM,GDIM)+COMMENT

#EXPERIMENT_DIR = os.getcwd() + '/'+ EXP_ROOT_DIR +'/' + EXPERIMENT_NAME
EXPERIMENT_DIR = './'+ EXP_ROOT_DIR +'/' + EXPERIMENT_NAME
CHECKPOINT_DIR = EXPERIMENT_DIR + '/checkpoint'

lib.print_model_settings(locals().copy())
def save_src(target_file_name):
    if os.path.exists(target_file_name):
        target_file_name += '.new.py'
    with open(sys.argv[0]) as src_file:
        with open(target_file_name,mode='w') as target:
            while True:
                s = src_file.readline()
                if len(s) > 0:
                    target.write(s)
                else:
                    break
if WORKMODE == 'train':
    if(os.path.isdir(EXPERIMENT_DIR)==False):
        os.makedirs(EXPERIMENT_DIR)
        
    if(os.path.isdir(CHECKPOINT_DIR)==False):
        os.makedirs(CHECKPOINT_DIR)
    save_src(EXPERIMENT_DIR + '/' + EXPERIMENT_NAME + '.py')
    
CHECKPOINT_NAME = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if CHECKPOINT_NAME is None:
    ITER_START = 0
else:
    model_file_name = os.path.split(CHECKPOINT_NAME)[1]
    ITER_START = int(model_file_name[model_file_name.rfind('-')+1:]) + 1
    

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator_first(n_samples, noise=None,name='Generator',zd=Z1DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, zd])

    output = lib.ops.linear.Linear('{}.Input'.format(name), zd, 4*4*4*GDIM, noise)
    output = lib.ops.batchnorm.Batchnorm('{}.BN1'.format(name), [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*GDIM, 4, 4])
    return output

def Generator_middle(n_samples,front_results,name='Generator'):
    output = lib.ops.deconv2d.Deconv2D('{}.2'.format(name), 4*GDIM, 2*GDIM, 5, front_results)
    output = lib.ops.batchnorm.Batchnorm('{}.BN2'.format(name), [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('{}.3'.format(name), 2*GDIM, GDIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('{}.BN3'.format(name), [0,2,3], output)
    output = tf.nn.relu(output)
    return output

def Generator_last(n_samples,middle_results,name='Generator'):
    output = lib.ops.deconv2d.Deconv2D('{}.5'.format(name), GDIM, 3, 5, middle_results)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def DCGenerator(n_samples, noise=None,name='Generator',zd=Z1DIM):
    front_results = Generator_first(n_samples, noise,name,zd)
    middle_results = Generator_middle(n_samples,front_results,name)
    return Generator_last(n_samples,middle_results,name)

def DCDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DDIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DDIM, 2*DDIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DDIM, 4*DDIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DDIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DDIM, 1, output)

    return tf.reshape(output, [-1])

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def Normalize(name, inputs):
    if ('Generator' in name):
        return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)    
    output = Normalize(name+'.N2', output)
    output = tf.nn.relu(output)            
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DDIM)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DDIM, output_dim=DDIM)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DDIM, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)    
    output = tf.nn.relu(output)            
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def ResGenerator(n_samples, noise=None,name='Generator',zd=Z1DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, zd])
    output = lib.ops.linear.Linear('{}.Input'.format(name), 128, 4*4*GDIM, noise)
    output = tf.reshape(output, [-1, GDIM, 4, 4])
    output = ResidualBlock('{}.1'.format(name), GDIM, GDIM, 3, output, resample='up')
    output = ResidualBlock('{}.2'.format(name), GDIM, GDIM, 3, output, resample='up')
    output = ResidualBlock('{}.3'.format(name), GDIM, GDIM, 3, output, resample='up')
    output = Normalize('{}.OutputN'.format(name), output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('{}.Output'.format(name), GDIM, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def ResDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DDIM, DDIM, 3, output, resample='down')
    output = ResidualBlock('Discriminator.3', DDIM, DDIM, 3, output, resample=None)
    output = ResidualBlock('Discriminator.4', DDIM, DDIM, 3, output, resample=None)
    output = tf.nn.relu(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DDIM, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    return output_wgan

Generator = eval('{}Generator'.format(ARCH))
Discriminator = eval('{}Discriminator'.format(ARCH))


def print_model_size(tag='',var_list=None):
    total_parameters = 0
    if var_list == None:
        # count all variables
        var_list = tf.trainable_variables()
    for variable in var_list:
        local_parameters = 1
        shape = variable.get_shape()  # getting shape of a variable
        for i in shape:
            local_parameters *= i.value  # mutiplying dimension values
        total_parameters += local_parameters
    string = tag + " param size: {:.6f}M \n".format(total_parameters / 1.0e6)
    print(string)
    if WORKMODE == 'train':
        with open(EXPERIMENT_DIR + '/' + 'info.txt',mode='a') as f:
            f.write(string + '\n')

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)


z1 = tf.random_normal([BATCH_SIZE, Z1DIM])
z2 = tf.random_uniform([BATCH_SIZE, N_SUBGEN],1e-8,1-1e-8)# sample uniform instead of applying invers norm CDF, for convinence

p_logits = tf.get_variable('p_logits',shape=[N_SUBGEN],dtype=tf.float32,initializer=tf.zeros_initializer())
p = tf.nn.softmax(p_logits)
F = tf.one_hot(tf.argmax(tf.log(p)-tf.log(-tf.log(z2)),axis=1),depth=N_SUBGEN)

#define model
#define model
#define model
#define model
#define model

def bias_fn(z1dim=100,N=10,delta=2):
    if delta < 0.02:
        return 0
    u = (2 * z1dim) ** 0.5 + delta * (1 - 1.0 / (8 * z1dim)) ** 0.5 - 1.0 / (8 * z1dim) ** 0.5
    hs = (u + (u ** 2 + delta * (32 * z1dim) ** 0.5) ** 0.5) ** 2 / 8.0 - z1dim
    b = -1.0 / N * (hs) ** 0.5
    return b

if GEN_TYPE == 'multigen':
    fake_datas_list = [Generator(BATCH_SIZE, z1, 'Generator{}'.format(i)) \
                       for i in range(N_SUBGEN)]

    fake_datas = tf.stack(fake_datas_list, axis=2)
    fake_data = tf.squeeze(tf.matmul(fake_datas, tf.expand_dims(F, axis=2)))
elif GEN_TYPE == 'vcgan':
    bias = bias_fn(Z1DIM,N_SUBGEN,DELTA)
    scale = -bias * N_SUBGEN or 1.0 #when DELTA=0, set scale=1.0, equivalent to common one-hot condition
    print("bias={:.4f}, scale={:.4f}".format(bias,scale))
    #    bias = 0
    #    scale = 1
    new_z = tf.concat([z1, F * scale + bias], 1)
    new_z_n = tf.transpose([tf.concat([z1, \
        tf.one_hot(np.ones((BATCH_SIZE), np.float) * i, depth=N_SUBGEN) * scale + bias],1) \
                            for i in range(N_SUBGEN)], (1, 2, 0))
    fake_data = Generator(BATCH_SIZE, new_z, 'Generator', zd=Z1DIM + N_SUBGEN)
elif GEN_TYPE == 'deligan':
    deligan_biases = tf.get_variable('deligan_biases',
                                     shape=[Z1DIM, N_SUBGEN], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-1, 1))
    deligan_scales_sqrt = tf.get_variable('deligan_scales_sqrt',
                                          shape=[Z1DIM, N_SUBGEN], dtype=tf.float32,
                                          initializer=tf.constant_initializer(DELIGAN_INIT_SCALE ** 0.5))

    deligan_scales = deligan_scales_sqrt ** 2 + 1e-8
    z1_copies = tf.transpose([z1 for i in range(N_SUBGEN)], [1, 2, 0])
    deligan_noise_n = deligan_scales * z1_copies + deligan_biases
    deligan_noise = tf.squeeze(tf.matmul(deligan_noise_n, tf.expand_dims(F, axis=2)))

    fake_data = Generator(BATCH_SIZE, deligan_noise, 'Generator')

disc_real = Discriminator(real_data)

disc_fake = Discriminator(fake_data)


EDGs = tf.placeholder(tf.float32,shape=[N_SUBGEN]) # [E[D(G_i(z))],]
p_loss = -tf.reduce_sum(p * EDGs)
global_step = tf.placeholder(tf.int32,shape=None)
p_lr = tf.train.exponential_decay(PLR,global_step=global_step,
                                  decay_steps=1000,decay_rate=PLRDR)
p_train_op = tf.train.GradientDescentOptimizer(p_lr).minimize(p_loss,var_list=[p_logits])
#p_train_op = tf.train.AdamOptimizer(PLR).minimize(p_loss,var_list=[p_sqrt])

if LRLD:
    LRx = tf.maximum(0., 1.-(tf.cast(global_step, tf.float32) / ITERS)) * LR
else:
    LRx = LR

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if GEN_TYPE == 'deligan':
    gen_params += [deligan_scales_sqrt ,deligan_biases]
#Todo: Here

if MODE == 'wgan-gp':
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
    gradient_penalty = tf.reduce_mean((tf.nn.relu(slopes-1.))**2) if RELU_GP else tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty
    gen_train_op = tf.train.AdamOptimizer(learning_rate=LRx, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=LRx, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
    disc_cost /= 2.
    gen_train_op = tf.train.AdamOptimizer(learning_rate=LRx, beta1=0.5).minimize(gen_cost,var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=LRx, beta1=0.5).minimize(disc_cost,var_list=disc_params)

# For generating samples
# fixed_noise_128 = tf.constant(np.random.normal(size=(128, Z1DIM)).astype('float32'))
N_SAMPLE_PER_SUBGEN = int(N_SUBGEN * np.ceil(8.0 / N_SUBGEN) ** 2)
N_TOTAL_SAMPLE = N_SAMPLE_PER_SUBGEN * N_SUBGEN
if CHECKPOINT_NAME is None:
    common_fixed_noise_ = np.random.normal(size=(N_SAMPLE_PER_SUBGEN, Z1DIM)).astype('float32')
    np.save(CHECKPOINT_DIR + '/common_fixed_noise',common_fixed_noise_)
else:
    common_fixed_noise_ = np.load(CHECKPOINT_DIR + '/common_fixed_noise.npy')
common_fixed_noise = tf.constant(common_fixed_noise_)

fixed_F = tf.one_hot(np.linspace(0, N_SUBGEN, num=N_TOTAL_SAMPLE, endpoint=False).astype('int32'), depth=N_SUBGEN)
if GEN_TYPE == 'multigen':
    fixed_fake_datas_list = [Generator(N_SAMPLE_PER_SUBGEN, common_fixed_noise, 'Generator{}'.format(i)) for i in
                             range(N_SUBGEN)]
    fixed_noise_samples = tf.stack(fixed_fake_datas_list, axis=0)
elif GEN_TYPE == 'vcgan':
    fixed_new_z = tf.concat([tf.constant(np.resize(common_fixed_noise_, \
                                                   (N_TOTAL_SAMPLE, Z1DIM))), fixed_F * scale + bias], 1)
    fixed_noise_samples = Generator(N_TOTAL_SAMPLE, fixed_new_z, 'Generator', zd=Z1DIM + N_SUBGEN)

elif GEN_TYPE == 'deligan':
    fixed_deligan_noise = tf.concat([common_fixed_noise_ * deligan_scales[:, i] + \
                             deligan_biases[:, i] for i in range(N_SUBGEN)], 0)

    fixed_noise_samples = Generator(Z1DIM, fixed_deligan_noise, 'Generator')


def generate_image(frame):
    samples = session.run(fixed_noise_samples)
    samples = ((samples + 1.) * (255. / 2)).astype('int32')
    lib.save_images.save_images(samples.reshape((N_TOTAL_SAMPLE, N_C, N_H, N_W)), \
                        EXPERIMENT_DIR + '/' + 'samples_{}.png'.format(frame))

#low-ram-usage version
def save_samples_to_npz(n_samples=50000, step=1,return_samples=False):
    all_samples = []
    for i in trange(int(np.ceil(n_samples * 1.0 / BATCH_SIZE))):
        all_samples.append(((session.run(fake_data) + 1.) * (255. / 2)).astype('uint8'))
    all_samples = np.concatenate(all_samples, axis=0)[:n_samples]
    all_samples = all_samples.reshape((-1, N_C, N_H, N_W)).transpose(0, 2, 3, 1)
    file_name = '{}_samples_iter{}'.format(n_samples,step)
    while True:
        try:
            #np.savez_compressed(EXPERIMENT_DIR + '/', all_samples)
            #saving to /tmp/ and then moving to dst may save time when dst is 
            #a network position
            np.savez_compressed('/tmp/' + EXPERIMENT_NAME + file_name,all_samples)
            ret = os.system('cp '+'/tmp/' + EXPERIMENT_NAME + file_name + 
                            '.npz '+EXPERIMENT_DIR + '/' + file_name + '.npz')
            if ret != 0:
                print("error in cp!")
                continue
            os.remove('/tmp/' + EXPERIMENT_NAME + file_name + '.npz')
            break
        except OSError as e:
            print("OSError when saving log, retry after 60s")
            print(e)
            time.sleep(60)
        except IOError as e:
            print("IOError when saving log, retry after 60s")
            print(e)
            time.sleep(60)    
    # to load the array,use "np.load('xxx.npz.)['arr_0']"
    if not return_samples:
        all_samples = None
    return all_samples        


def get_inception_score(iters, samples):
    score_and_std = lib.inception_score.get_inception_score( \
        list(samples), bs=16)
    with open(EXPERIMENT_DIR + '/' + 'inception_score.txt', mode='a') as f:
        f.write('{} {} {}\n'.format(iters, score_and_std[0], score_and_std[1]))
    return score_and_std


def make_dataset_gen(ds):
    p_list = []
    gen_list = []

    def inf_train_gen(name, gen):
        while True:
            for images in gen():
                yield images[0].reshape([BATCH_SIZE, -1])

    for name, p, path in ds:
        if p != 0:
            if name == 'cifar':
                train_gen, _ = lib.cifar10.load(BATCH_SIZE, data_dir=path)
            else:
                train_gen, _ = dataloader.load(BATCH_SIZE, data_dir=path,\
                                       imsize=N_H,validation_ratio=0)
            gen_list.append(inf_train_gen(name, train_gen))
            p_list.append(p)

    def inf_train_gen_mix():
        while True:
            count = np.random.multinomial(BATCH_SIZE, p_list)
            for i in range(1, len(count)):
                count[i] += count[i - 1]
            count = [0] + list(count)
            #            print(count)
            image_list = []
            for i, gen in enumerate(gen_list):
                images = gen.next()[count[i]:count[i + 1]]
                image_list.append(images)
            images = np.concatenate(image_list)
            yield images

    return inf_train_gen_mix

print_model_size(tag='gen',var_list=gen_params)
print_model_size(tag='dis',var_list=disc_params)
print_model_size(tag='all_params')

# Train loop
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())

if WORKMODE == 'train':
    inf_train_gen = make_dataset_gen(DATASETS)
    gen = inf_train_gen()
    #save a batch of real images
    real_samples = gen.next().astype('int32').reshape((BATCH_SIZE, N_C, N_H, N_W))
    lib.save_images.save_images(real_samples,EXPERIMENT_DIR+'/real_samples.png')
saver = tf.train.Saver(max_to_keep=None)
if(CHECKPOINT_NAME!=None):
    saver.restore(session,CHECKPOINT_NAME)
    lib.plot.load(log_dir=EXPERIMENT_DIR,iter_start=ITER_START)
    print("model restroed from " + CHECKPOINT_NAME)

if WORKMODE == 'train':
    for iteration in trange(ITER_START, ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op,{global_step:iteration})
        
        #print deligan debug info
        if GEN_TYPE == 'deligan' and (iteration % 250 == 249 or iteration - ITER_START < 5):
            deligan_scales_, deligan_biases_ = session.run([deligan_scales, deligan_biases])

            print('mean_deligan_scales = {}' \
                .format(np.mean(deligan_scales_, axis=0).astype('float64').round(4)))
            if iteration == ITER_START:
                deligan_scales_old, deligan_biases_old = deligan_scales_, deligan_biases_
            else:
                deligan_scales_change = np.abs(deligan_scales_old - deligan_scales_).sum()
                deligan_biases_change = np.abs(deligan_biases_old - deligan_biases_).sum()
                print('deligan_scales_change = {:.6f},  deligan_biases_change = {:.6f}' \
                    .format(deligan_scales_change, deligan_biases_change))
                deligan_scales_old, deligan_biases_old = deligan_scales_, deligan_biases_

        # Train critic
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
            
        if TRAIN_P_PERIOD > 0:# if trainable categorical probabilities
            EDG_j_list = []
            F_batch_list = []
            disc_fake_batch_list = []

        for i in xrange(disc_iters):
            if not TRAIN_P_PERIOD > 0: # if fixed categorical probabilities
                _data = gen.next()
                _disc_cost, _ = session.run([disc_cost, disc_train_op],
                        feed_dict={real_data_int: _data,global_step:iteration})
            else: # if trainable categorical probabilities
                _data = gen.next()
                _disc_cost, _,F_,disc_fake_ = session.run(
                        [disc_cost, disc_train_op,F,disc_fake], 
                        feed_dict={real_data_int: _data,global_step:iteration})
                F_batch_list.append(F_)
                disc_fake_batch_list.append(disc_fake_)
        lib.plot.plot('train disc cost', _disc_cost)

        # learn categorical probabilities
        if TRAIN_P_PERIOD > 0:
            if iteration % TRAIN_P_PERIOD == TRAIN_P_PERIOD - 1 and iteration >= TRAIN_P_START:
                F_batch_list = np.concatenate(F_batch_list)
                disc_fake_batch_list = np.concatenate(disc_fake_batch_list)
                for j in range(N_SUBGEN):
                    count_j = np.sum(F_batch_list[:,j])
                    if count_j == 0:
                        print('WARNING: gen_{} seems dead!'.format(j))
                        count_j += 1e-3  #avoid zero division error
                    EDG_j = np.sum(disc_fake_batch_list * F_batch_list[:,j]) / (count_j)
                    EDG_j_list.append(EDG_j)
                _, p_,p_lr_ = session.run([p_train_op, p,p_lr],
                            feed_dict={EDGs: EDG_j_list,global_step:iteration})
                if iteration % 50 == 0:
                    print('EDG_j_list = {},p_lr={:.5f}'.format(EDG_j_list,p_lr_))
                    print('p_ = {}'.format(p_))
            else:
                p_ = session.run(p)
            for i, p_i in enumerate(p_):
                lib.plot.plot('p_{}'.format(i), p_i)
        
        # if (ENABLE_DIS_CLS_LOSS or ENABLE_GEN_CLS_LOSS):
        #     lib.plot.plot('classifier cost', _cls_loss)
        lib.plot.plot('time', time.time() - start_time)
        if iteration % SAVE_SAMPLES_PERIOD == SAVE_SAMPLES_PERIOD - 1:
            samples_50k = save_samples_to_npz(50000, iteration)
            if CALC_INCEPTION_SCORE:
                inception_score = get_inception_score(iteration, samples_50k)
                lib.plot.plot('inception score', inception_score[0])

        if iteration % PREVIEW_PERIOD == PREVIEW_PERIOD - 1:
            generate_image(iteration)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 500 == 499):
            lib.plot.flush(log_dir=EXPERIMENT_DIR,header=EXPERIMENT_NAME)

        if (iteration % SAVE_CHECKPOINT_PERIOD == (SAVE_CHECKPOINT_PERIOD - 1)):
            try:
                while True:
                    saver.save(session, CHECKPOINT_DIR + "/my_model", \
                               global_step=iteration)
                    break
            except OSError as e:
                print("OSError when saving log, retry after 60s")
                print(e)
                time.sleep(60)
            except IOError as e:
                print("IOError when saving log, retry after 60s")
                print(e)
                time.sleep(60)    

        lib.plot.tick()
        if TRAIN_P_PERIOD > 0 and iteration == ITERS - 1: # log last p to txt
            with open(EXPERIMENT_DIR + '/' + 'p_{:.4f}.txt'\
                      .format(p_.min()),mode='a') as f:
                for i,p_i in enumerate(p_):
                    f.write('p{}\t{:.8f}\n'.format(i,p_i))
    print('Training finished!')
else:  # WORKMODE == 'generate'
    generate_image('final')
    print('Generating finished!')

