# Virtual Conditional Generative Adversarial Networks
> This anonymous repo is a TensorFlow implementation of paper-"Virtual Conditional Generative Adversarial Networks",  
> which is submitted to ICML-2019.

---

## Requirements
> python version: python2.7  
> CUDA9: install tensorflow with `pip install tensorflow-gpu==1.12`
> CUDA10: install tensorflow according to this [issue][6]
> run `pip install -r requirements`

## Datasets
> * MNIST: In this repo, images are preprocessed into 32*32. You can download it [here][1]  
> * Fashion MNIST: All images in Fashion MNIST are also preprocessed into 32*32. You can  
> download it [here][2] 
> * CIFAR-10: Download [CIFAR-10][3] (Python version) and extract files into `HOME_DIR/datasets/  
> cifar-10-batches-py/` 
> * CelebA: Get CelebA and crop according to [DCGAN][4] 
> * Cartoon Set: Download [Cartoon Set 100K][5], extract and move all images to one folder and use `cartoon_preprocessing.py` to preprocess  
> images

> In default setting, all the datasets locate in `HOME_DIR/datasets/DATASET/`, you can change  
> the `HOME_DIR` in `vcgan_32x32.py` and `vcgan_64x64.py` to make sure codes  
> could run correctly.

## Train and Generate

### MNIST, Fashion MNIST and CIFAR-10
> choose dataset:set ratio of each dataset in `DATASETS`, for example, if you want to train on  
> MNIST, set
```
DATASETS =  [#name, ratio, path
            ['cifar',   0.0,home_dir + '/datasets/cifar-10-batches-py/'],
            ['mnist32', 1.0,home_dir + '/datasets/mnist32/'],
            ['fashion32', 0.0,home_dir + '/datasets/fashion_mnist32/'],
            ]
```
> vcGAN-FP(fixed p):set `TRAIN_P_PERIOD = 0`, and run `python vcgan_32x32.py`  
> vcGAN-LP(learnable p):set `TRAIN_P_PERIOD = 1`, and run `python vcgan_32x32.py`
> WGAN-GP baseline: set GEN_TYPE = 'multigen'; Z2DIM = N_SUBGEN = 1; and run `python vcgan_32x32.py`

### CelebA and Cartoon Set
> choose dataset:set ratio of each dataset in `DATASETS`, for example, if you want to train on  
> Ce7Ca, set
```
DATASETS =  [#name, ratio, path
            ['celeba',  0.7,home_dir + '/datasets/celebA/'],
            ['cartoon', 0.3,home_dir + '/datasets/cartoonset100k_small/'],
            ]
```
> vcGAN-FP(fixed p):set `TRAIN_P_PERIOD = 0`, and run `python vcgan_64x64.py`  
> vcGAN-LP(learnable p):set `TRAIN_P_PERIOD = 1`, and run `python vcgan_64x64.py`

## FID and Inception Score
> set `ROOT_EXP_DIR` be the output directory of your model,e.g. exp_output/  
> FID: run `python calc_fid_from _npz.py`  
> Inception Score: run `python calc_inception_score_from_npz.py`  

## NOTES
> A lot of arguments could be modified in `vcgan_32x32.py` and `vcgan_64x64.py`. You can change  
> these arguments to see how this model performs in other settings.

[1]:https://www.dropbox.com/s/m09wssvhwe790w1/mnist32.zip?dl=0
[2]:https://www.dropbox.com/s/3vs30y3me0hqp3e/fashion_mnist32.zip?dl=0
[3]:https://www.cs.toronto.edu/~kriz/cifar.html
[4]:https://github.com/soumith/dcgan.torch
[5]:https://google.github.io/cartoonset/
[6]:https://github.com/tensorflow/tensorflow/issues/22706
