import os
import os.path as osp
import json
import math

import numpy as np

import chainer
from chainer import cuda, Variable
import chainer.functions as F

def softmax_sample(p):
    """
    input p: softmax output, chainer.Variable, batchsize * num_class
    output: sampled index
    """
    xp = cuda.get_array_module(p.data)
    rand = xp.random.uniform(size=p.data.shape[0], dtype=p.data.dtype)
    next_state = xp.zeros(p.data.shape[0], dtype="int32")
    xp.ElementwiseKernel(
        'raw T p, raw T rand, I num',
        'I next_state',
        '''
            T cumsum = 0;
            for(I j=0; j < num; j++) {
                cumsum += p[i * num + j];
                if(cumsum > rand[i]) {
                    next_state = j;
                    break;
                }
            }
        ''',
        'sample')(p.data, rand, p.data.shape[1], next_state)
    return next_state

def calc_max_ind(seq):
    length = np.array([seq.shape[1]-1]*seq.shape[0])
    for ind, s in enumerate(seq):
        for i, w in enumerate(s):
            if w==0:
                length[ind] = i
                break
    return length

def beam_search(model, vis_feats, beam_width):
    xp = cuda.get_array_module(vis_feats)
    results = []
    for b in range(vis_feats.shape[0]):
        model.LSTM_initialize()
        candidates = [(model, [model.vocab_size+1], 0, 0)]
        feat = vis_feats[b][xp.newaxis,:]
        for i in range(model.seq_length):
            next_candidates = []
            for prev_net, tokens, sum_likelihood, likelihood in candidates:
                if tokens[-1] == 0:
                    next_candidates.append((None, tokens, sum_likelihood, likelihood))
                    continue
                net = prev_net.copy()
                w = Variable(xp.asarray([tokens[-1]]).astype(np.int32))
                h = net.forward(feat, w, i)
                token_likelihood = cuda.to_cpu(F.log_softmax(h).data[:,:-1])[0]
                order = token_likelihood.argsort()[:-beam_width-1:-1]
                next_candidates.extend([(net, tokens + [j], sum_likelihood+token_likelihood[j], (likelihood * len(tokens) + token_likelihood[j])/(len(tokens) + 1)) for j in order])
            candidates = sorted(next_candidates, key=lambda x: -x[3])[:beam_width]
            if all([candidate[1][-1] == 0 for candidate in candidates]):
                break
        result = [{'sent':[int(w) for w in candidate[1][1:-1]],'ppl':float(math.exp(-candidate[3]))} for candidate in candidates]
        results.append(result)
    return results

def make_graph(ve, cca, loader, split, params, xp):
    target_save_dir = osp.join(params['save_dir'], 'prepro', params['dataset']+'_'+params['splitBy'])
    graphs = []
    with chainer.using_config('train', False):
        loader.resetImageIterator(split)
        while True:
            data = loader.getImageBatch(split, params)
            img_ann_ids = data['img_ann_ids']
            image_id = data['image_id']
            feats = Variable(xp.array(data['feats'], dtype=xp.float32))
            vis_enc_feats = ve(feats)
            vis_enc_feats = cca.vis_forward(vis_enc_feats)
            score = cuda.to_cpu(F.matmul(vis_enc_feats, vis_enc_feats, transb=True).data)
            float_score = []
            for one_score in score:
                float_score.append([float(one) for one in one_score])
            graphs.append({'image_id':image_id, 'ann_ids':img_ann_ids, 'cossim':float_score})
            print('{}/{}'.format(data['bounds']['it_pos_now'], data['bounds']['it_max']))
            if data['bounds']['wrapped']:
                break
    graph_path = osp.join(target_save_dir, params['id']+'_graphs.json')
    with open(graph_path, 'w') as f:
        json.dump(graphs, f)
    loader.load_graph(graph_path)

def load_vcab_init(dictionary, save_path, 
                   glove_path='/data/unagi0/mtanaka/wordembedding/glove/glove.840B.300d.txt'):
    if not os.path.exists(save_path):
        initial_emb = np.zeros((len(dictionary)+2, 300), dtype = np.float32)
        word2emb = {}
        with open(glove_path, 'r') as f:
            entries = f.readlines()
            for entry in entries:
                vals = entry.split(' ')
                word = vals[0]
                vals = list(map(float, vals[1:]))
                word2emb[word] = np.array(vals)
            for word in list(dictionary.keys()):
                if word not in word2emb:
                    continue
                initial_emb[int(dictionary[word]), :300] = word2emb[word]
            np.save(save_path, initial_emb)
    else:
        initial_emb = np.load(save_path)
    return initial_emb