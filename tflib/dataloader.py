import numpy as np
import scipy.misc
import time
from os import listdir
from os.path import isfile, join
def make_generator(path,batch_size,is_validation_set=False,\
                   validation_ratio=0.2,imsize=64):
    files = [f for f in listdir(path) if isfile(join(path, f))]
#    files = files[:1000] #subset
    if(is_validation_set==True):
        files=files[:int(validation_ratio*len(files))]
    else:
        files=files[int(validation_ratio*len(files)):]
    image_count=len(files)
    if(image_count==0):
        raise Exception("no image found in "+path)
    
    print("image count = "+str(image_count))
    
    epoch_count = [1]
    def get_epoch():
        
        images = np.zeros((batch_size, 3, imsize, imsize), dtype='uint8')
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, f_name in enumerate(files):
            image = scipy.misc.imread(path+f_name)
            if len(image) != imsize:
                image = scipy.misc.imresize(image,[imsize,imsize],interp='bicubic')
            images[n % batch_size] = image.transpose(2,0,1)
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
