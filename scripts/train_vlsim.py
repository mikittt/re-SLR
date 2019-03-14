import argparse
import os
import os.path as osp
import math
import numpy as np
import sys
sys.path.append('./')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable, optimizers, serializers

from misc.DataLoader import DataLoader
from misc.utils import calc_max_ind, load_vcab_init
from misc.eval_utils import compute_margin_loss, calc_rank_loss, calc_rank_acc
from models.base import VisualEncoder, LanguageEncoder, LanguageEncoderAttn, MetricNet
from models.Reinforcer import ListenerReward
import config

def train_vl(params):
    target_save_dir = osp.join(params['save_dir'], 'prepro', params['dataset']+'_'+params['splitBy'])
    graph_dir = osp.join('log_graph', params['dataset']+'_'+params['splitBy'])
    model_dir = osp.join(params['save_dir'], 'model', params['dataset']+'_'+params['splitBy'])
    if not osp.isdir(graph_dir):
        os.makedirs(graph_dir)
    if not osp.isdir(model_dir):
        os.makedirs(model_dir)
        
    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['image_feats'] = 'old'+params['image_feats']
        params['ann_feats'] = 'old'+params['ann_feats']
        params['id'] = 'old'+params['id']
        params['word_emb_path'] = 'old'+params['word_emb_path']
        
        
    loader = DataLoader(params)
    
    featsOpt = {'ann':osp.join(target_save_dir, params['ann_feats']),
                'img':osp.join(target_save_dir, params['image_feats'])}
    loader.loadFeats(featsOpt) 
    loader.shuffle('train')
    
    # model setting
    batch_size = params['batch_size']
    gpu_id = params['gpu_id']
    seq_per_ref = params['seq_per_ref']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
    
    ve = VisualEncoder(res6=L.ResNet152Layers().fc6).to_gpu(gpu_id)
    if 'attention' in params['id']:
        print('attention language encoder')
        le = LanguageEncoderAttn(len(loader.ix_to_word))
        save_model = ListenerReward(len(loader.ix_to_word), attention=True)
    else:
        le = LanguageEncoder(len(loader.ix_to_word))
        save_model = ListenerReward(len(loader.ix_to_word), attention=False)
    
    if params['pretrained_w']:
        print('pretrained word embedding...')
        word_emb = load_vcab_init(loader.word_to_ix, osp.join(target_save_dir,params['word_emb_path']))
        le.word_emb.W.data = word_emb
    le.to_gpu(gpu_id)
    me = MetricNet().to_gpu(gpu_id)
    
    ve_optim = optimizers.Adam(alpha=4e-5, beta1=0.8)
    le_optim = optimizers.Adam(alpha=4e-4, beta1=0.8)
    me_optim = optimizers.Adam(alpha=4e-4, beta1=0.8)
    ve_optim.setup(ve)
    le_optim.setup(le)
    me_optim.setup(me)
    
    ve_optim.add_hook(chainer.optimizer.GradientClipping(0.1))
    le_optim.add_hook(chainer.optimizer.GradientClipping(0.1))
    me_optim.add_hook(chainer.optimizer.GradientClipping(0.1))
    ve.joint_enc.W.update_rule.hyperparam.alpha = 4e-4
    ve.joint_enc.b.update_rule.hyperparam.alpha = 4e-4

    iteration = 0
    epoch = 0
    val_loss_history = []
    val_acc_history = []
    val_rank_acc_history = []
    min_val_loss = 100
    max_acc = 0
    while True:
        chainer.config.train = True
        chainer.config.enable_backprop = True
        ve.zerograds()
        le.zerograds()
        me.zerograds()
        data = loader.getBatch('train', params)
        ref_ann_ids = data['ref_ann_ids']
            
        pos_feats = Variable(xp.array(data['feats'], dtype=xp.float32))
        neg_feats = Variable(xp.array(data['neg_feats'], dtype=xp.float32))
        feats = F.concat([pos_feats, neg_feats, pos_feats], axis=0)
            
        seqz  = np.concatenate([data['seqz'], data['seqz'], data['neg_seqz']], axis=0)
        lang_last_ind = calc_max_ind(seqz)
        seqz = Variable(xp.array(seqz, dtype=xp.int32))
        labels = Variable(xp.concatenate([xp.ones((batch_size*seq_per_ref)),
                                          xp.zeros((batch_size*seq_per_ref)), xp.zeros((batch_size*seq_per_ref))]).astype(xp.int32))
        
        vis_enc_feats = ve(feats)
        lang_enc_feats = le(seqz, lang_last_ind)
        score = me(vis_enc_feats, lang_enc_feats).reshape(labels.shape)
        
        loss = F.sigmoid_cross_entropy(score, labels)
        loss.backward()
        ve_optim.update()
        le_optim.update()
        me_optim.update()
            
        if data['bounds']['wrapped']:
            print('{} epoch finished!'.format(epoch))
            loader.shuffle('train')
            epoch+=1
            
        if iteration %params['losses_log_every']==0:
            print('{} iter ({} epoch): train loss {}'.format(iteration, epoch, loss.data))
            
        ## validation
        if (iteration % params['save_checkpoint_every'] == 0 and iteration >0):
            chainer.config.train = False
            chainer.config.enable_backprop = False
            loader.resetImageIterator('val')
            loss_sum = 0
            loss_evals = 0
            accuracy = 0
            rank_acc = 0
            rank_num = 0
            while True:
                data = loader.getImageBatch('val', params)
                image_id = data['image_id']
                img_ann_ids = data['img_ann_ids']
                sent_ids = data['sent_ids']
                gd_ixs = data['gd_ixs']
                feats = Variable(xp.array(data['feats'], dtype=xp.float32))
                seqz = data['seqz']
                scores = []
                for i, sent_id in enumerate(sent_ids):
                    ## image内の全ての候補領域とscoreを算出する
                    gd_ix = gd_ixs[i]
                    labels = xp.zeros(len(img_ann_ids),dtype=xp.int32)
                    labels[gd_ix] = 1
                    labels = Variable(labels)

                    sent_seqz = np.concatenate([[seqz[i]] for _ in range(len(img_ann_ids))], axis=0)
                    lang_last_ind =  calc_max_ind(sent_seqz)
                    sent_seqz = Variable(xp.array(sent_seqz, dtype=xp.int32))

                    vis_enc_feats = ve(feats)
                    lang_enc_feats = le(sent_seqz, lang_last_ind)
                    score = me(vis_enc_feats, lang_enc_feats).reshape(labels.shape)
                    loss = F.sigmoid_cross_entropy(score, labels)
                    scores.append(score[gd_ix].data)

                    loss_sum += loss.data
                    loss_evals += 1
                    _, pos_sc, max_neg_sc = compute_margin_loss(score.data, gd_ix, 0)
                    if pos_sc > max_neg_sc:
                        accuracy += 1
                
                if params['dataset']=='refgta':
                    rank_a, rank_n = calc_rank_acc(scores, data['rank'])
                    rank_acc += rank_a
                    rank_num += rank_n 
                print('{} iter | {}/{} validating acc : {}'.format(iteration, data['bounds']['it_pos_now'], data['bounds']['it_max'], accuracy/loss_evals))
                
                if data['bounds']['wrapped']:
                    print('validation finished!')
                    fin_val_loss = cuda.to_cpu(loss_sum/loss_evals)
                    fin_val_acc = accuracy/loss_evals
                    break
            val_loss_history.append(fin_val_loss)
            val_acc_history.append(fin_val_acc)
            if min_val_loss>fin_val_loss:
                print('val loss {} -> {} improved!'.format(min_val_loss, val_loss_history[-1]))
                min_val_loss = fin_val_loss
                
            if max_acc<fin_val_acc:
                max_acc = fin_val_acc
                save_model.ve = ve
                save_model.le = le
                save_model.me = me
                serializers.save_hdf5(osp.join(model_dir, params['id']+".h5"), save_model)
                
            ## graph
            plt.title("accuracy")
            plt.plot(np.arange(len( val_acc_history)),  val_acc_history, label="val_accuracy")
            plt.legend()
            plt.savefig(os.path.join(graph_dir, params['id'] + "_acc.png"))
            plt.close()

            plt.title("loss")
            plt.plot(np.arange(len(val_loss_history)), val_loss_history, label="val_loss")
            plt.legend()
            plt.savefig(os.path.join(graph_dir, params['id'] + "_loss.png"))
            plt.close()
        
            if params['dataset'] == 'refgta':
                print(rank_num)
                val_rank_acc_history.append(rank_acc/rank_num)
                plt.title("rank loss")
                plt.plot(np.arange(len(val_rank_acc_history)), val_rank_acc_history, label="rank_acc")
                plt.legend()
                plt.savefig(os.path.join(graph_dir, params['id'] + "_rank_acc.png"))
                plt.close()
        
        # learning rate decay
        if iteration > params['learning_rate_decay_start'] and params['learning_rate_decay_start'] >= 0:
            frac = (iteration - params['learning_rate_decay_start']) / params['learning_rate_decay_every']
            decay_factor = math.pow(0.1, frac)
            ve_optim.alpha *= decay_factor
            le_optim.alpha *= decay_factor
            me_optim.alpha *= decay_factor
            
        iteration+=1
    

# python scripts/train_vlsim.py -g 0 -id lm
if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    train_vl(params)