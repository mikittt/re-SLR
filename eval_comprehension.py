import argparse
import os.path as osp
import numpy as np

import chainer
from chainer import Variable, cuda, serializers

from misc.DataLoader import DataLoader
from misc.utils import calc_max_ind
from models.base import VisualEncoder, LanguageEncoder, LanguageEncoderAttn
from models.Listener import CcaEmbedding
from models.LanguageModel import vis_combine, LanguageModel
from misc.eval_utils import compute_margin_loss, computeLosses
import config

def eval_all(params):
    target_save_dir = osp.join(params['save_dir'],'prepro', params['dataset']+'_'+params['splitBy'])
    model_dir = osp.join(params['save_dir'],'model', params['dataset']+'_'+params['splitBy'])
    
    if params['old'] and params['dataset'] in ['refcoco','refcoco+','refcocog']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['image_feats'] = 'old'+params['image_feats']
        params['ann_feats'] = 'old'+params['ann_feats']
        params['id'] = 'old'+params['id']
        
    loader = DataLoader(params)
    
    featsOpt = {'ann':osp.join(target_save_dir, params['ann_feats']),
                'img':osp.join(target_save_dir, params['image_feats'])}
    loader.loadFeats(featsOpt) 
    chainer.config.train = False
    chainer.config.enable_backprop = False
    
    gpu_id = params['gpu_id']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
    
    ve = VisualEncoder().to_gpu(gpu_id)
    if 'attention' in params['id']:
        print('attn')
        le = LanguageEncoderAttn(len(loader.ix_to_word)).to_gpu(gpu_id)
    else:
        le = LanguageEncoder(len(loader.ix_to_word)).to_gpu(gpu_id)
    cca = CcaEmbedding().to_gpu(gpu_id)
    lm  = LanguageModel(len(loader.ix_to_word), loader.seq_length).to_gpu(gpu_id)
    
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"ve.h5"), ve)
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"le.h5"), le)
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"cca.h5"), cca)
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"lm.h5"), lm)
    
    accuracy = 0
    loss_evals  = 0
    while True:
        data = loader.getImageBatch(params['split'], params)
        image_id = data['image_id']
        img_ann_ids = data['img_ann_ids']
        sent_ids = data['sent_ids']
        gd_ixs = data['gd_ixs']
        feats = Variable(xp.array(data['feats'], dtype=xp.float32))
        seqz = data['seqz']
        lang_last_ind = calc_max_ind(seqz)
        for i, sent_id in enumerate(sent_ids):
            gd_ix = gd_ixs[i]
            labels = xp.zeros(len(img_ann_ids), dtype=xp.int32)
            labels[gd_ix] = 1
            labels = Variable(labels)

            sent_seqz = np.concatenate([[seqz[i]] for _ in range(len(img_ann_ids))],axis=0)
            one_last_ind =  np.array([lang_last_ind[i]]*len(img_ann_ids))
            sent_seqz = Variable(xp.array(sent_seqz, dtype=xp.int32))
                
            vis_enc_feats = ve(feats)
            lang_enc_feats = le(sent_seqz, one_last_ind)
            cossim, vis_emb_feats = cca(vis_enc_feats, lang_enc_feats)
            vis_feats = vis_combine(vis_enc_feats, vis_emb_feats)
            logprobs = lm(vis_feats, sent_seqz, one_last_ind).data
            
            lm_scores = -computeLosses(logprobs, one_last_ind)  
            
            if params['mode']==0:
                _, pos_sc, max_neg_sc = compute_margin_loss(lm_scores, gd_ix, 0)
            elif params['mode']==1:
                _, pos_sc, max_neg_sc = compute_margin_loss(cossim.data, gd_ix, 0)
            elif params['mode']==2:
                scores = cossim.data + params['lamda'] * lm_scores
                _, pos_sc, max_neg_sc = compute_margin_loss(scores, gd_ix, 0)
            if pos_sc > max_neg_sc:
                accuracy += 1
            loss_evals += 1
            print('{}-th: evaluating [{}]  ... image[{}/{}] sent[{}], acc={}'.format(loss_evals, params['split'], data['bounds']['it_pos_now'], data['bounds']['it_max'], i, accuracy*100.0/loss_evals))
        
        if data['bounds']['wrapped']:
            print('validation finished!')
            f = open('result/'+params['dataset']+params['split']+params['id']+str(params['mode'])+str(params['lamda'])+'comp.txt', 'w') # 書き込みモードで開く
            f.write(str(accuracy*100.0/loss_evals)) # 引数の文字列をファイルに書き込む
            f.close() 
            break
                    
if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    eval_all(params)