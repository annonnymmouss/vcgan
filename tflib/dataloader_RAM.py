# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
import time
from os import listdir
from os.path import isfile, join
import tqdm
def make_generator(path,batch_size,is_validation_set=False,\
                   validation_ratio=0.2,imsize=64):
    epoch_count = [1]
    images = np.zeros((batch_size, 3, imsize, imsize), dtype='int32')
    random_state = np.random.RandomState(epoch_count[0])
    cache_name = '/tmp/{}_{}_{}_{}.npy'.format(path.replace('/','_'),\
            is_validation_set,validation_ratio,imsize)
    try:
        all_images_in_RAM = np.load(cache_name)
        image_count= len(all_images_in_RAM)
        print('load {} images from cache {}'.format(image_count,cache_name))
    except:
        print('cache {} not found.'.format(cache_name))
        files = [f for f in listdir(path) if isfile(join(path, f))]
#        files = files[:1000] #subset
        if(is_validation_set==True):
            files=files[:int(validation_ratio*len(files))]
        else:
            files=files[int(validation_ratio*len(files)):]
        random_state.shuffle(files)
        image_count=len(files)
        if(image_count==0):
            raise Exception("no image found in "+path)
        
        print("image count = "+str(image_count))
        
        all_images_in_RAM=np.zeros((image_count,3,imsize,imsize),dtype='uint8')
        t0=time.time()
        for n in tqdm.trange(image_count):
            f_name = files[n]
            image = scipy.misc.imread(path+f_name)
            image = image[:,:,:3]#if RGBA, convert to RGB
            if len(image) != imsize:
                image = scipy.misc.imresize(image,[imsize,imsize],interp='bicubic')
            all_images_in_RAM[n] = image.transpose(2,0,1)
        print("Loading "+str(image_count) + " images to RAM costs "\
              + str(time.time()-t0)[:8] + "s") 
        print('caching images to '+cache_name)
        np.save(cache_name,arr=all_images_in_RAM)
    def get_epoch():
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(all_images_in_RAM)
        epoch_count[0] += 1
        for n in range(image_count):
            images[n % batch_size] = all_images_in_RAM[n]
            if (1+n) % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size,data_dir,\
                         validation_ratio=0.2,imsize=64):
    if validation_ratio != 0:
        return (
            make_generator(data_dir,batch_size,False,validation_ratio,imsize),
            make_generator(data_dir,batch_size,True,validation_ratio,imsize)
        )
    else:
        return (
            make_generator(data_dir,batch_size,False,validation_ratio,imsize),
            None
        )

if __name__ == '__main__':
    train_gen, valid_gen = load(64,'~/datasets/mnist32/')
    average_time_per_batch=0.0
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        t=time.time() - t0
        average_time_per_batch=(average_time_per_batch*(i-1)+t)/i
        print("average time to load a batch: {}s\tthis batch cost {}s".\
              format(str(average_time_per_batch)[:6],str(t)[:6]))
        if i == 1000:
            break
        t0 = time.time()
