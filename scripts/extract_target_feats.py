import argparse
import os
import numpy as np
import sys
import os.path as osp
sys.path.append('./')

from PIL import Image
from tqdm import tqdm
from misc.DataLoader import DataLoader
import chainer
from chainer import cuda, Variable
import chainer.links as L

import config

def extract_feature(params):
    if params['dataset'] in ['refcoco', 'refcoco+', 'refcocog']:
        image_root =  params['mscoco_root']
    elif params['dataset'] == 'refgta':
        image_root = params['gta_root']
    target_save_dir = osp.join(params['save_dir'],'prepro', params['dataset']+'_'+params['splitBy'])
    
    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['ann_feats'] = 'old'+params['ann_feats']
        
    loader = DataLoader(params)
        
    # model setting
    batch_size = params['batch_size']
    gpu_id = params['gpu_id']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
    
    res = L.ResNet152Layers()
    res.to_gpu(gpu_id)
    chainer.config.train = False
    chainer.config.enable_backprop = False
    
    anns = loader.anns
    images = loader.Images
    perm = np.arange(len(anns))
    ann_feats = []
    for bs in tqdm(range(0, len(anns), batch_size)):
        batch = []
        for ix in perm[bs:bs+batch_size]:
            ann = anns[ix]
            h5_id = ann['h5_id']
            assert h5_id==ix, 'h5_id not match' 
            img = images[ann['image_id']]
            x1, y1, w, h = ann['box']
            image = Image.open(os.path.join(image_root, img['file_name'])).convert('RGB')
            if h<=w:
                nh, nw = int(224/w*h), 224
            else:
                nh, nw = 224, int(224/h*w)
            image = image.crop((x1, y1, x1+w, y1+h)).resize((nw, nh), Image.ANTIALIAS)
            image = np.array(image).astype(np.float32)[:, :, ::-1]
            image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
            image = image.transpose((2, 0, 1))
            pad_image = np.zeros((3,224,224), dtype=np.float32)
            if nh<=nw:
                pad_image[:,(224-nh)//2:(224-nh)//2+nh,:] = image
            else:
                pad_image[:,:,(224-nw)//2:(224-nw)//2+nw] = image
            batch.append(pad_image)
        batch = Variable(xp.array(batch, dtype=xp.float32))
        feature = res(batch, layers=['pool5'])
        feature = cuda.to_cpu(feature['pool5'].data)
        ann_feats.extend(feature)
    np.save(os.path.join(target_save_dir, params['ann_feats']), ann_feats)



if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    extract_feature(params)