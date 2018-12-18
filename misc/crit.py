import chainer.functions as F
from chainer import cuda
from chainer import Variable

def emb_crits(emb_flows, margin, vlamda=1, llamda=1):
    xp = cuda.get_array_module(emb_flows['vis'][0])
    batch_size = emb_flows['vis'][0].shape[0]
    
    zeros = Variable(xp.zeros(batch_size, dtype=xp.float32))
    vis_loss = F.mean(F.maximum(zeros, margin+emb_flows['vis'][1]-emb_flows['vis'][0]))
    lang_loss = F.mean(F.maximum(zeros, margin+emb_flows['lang'][1]-emb_flows['lang'][0]))
    return vlamda*vis_loss + llamda*lang_loss
    
def lm_crits(lm_flows, num_labels, margin, vlamda=1, llamda=0, langWeight=1):
    xp = cuda.get_array_module(lm_flows['T'])
    ## language loss
    n = 0
    lang_loss = 0
    Tprob = lm_flows['T']
    lang_num = num_labels['T']
    lang_loss -= F.sum(Tprob)/(sum(lang_num)+len(lang_num))
    if vlamda==0 and llamda==0:
        return lang_loss
    
    def triplet_loss(flow, num_label):
        pairGenP = flow[0]
        unpairGenP = flow[1]
        zeros = Variable(xp.zeros(pairGenP.shape[1], dtype=xp.float32))
        pairSentProbs = F.sum(pairGenP,axis=0)/(num_label+1)
        unpairSentProbs = F.sum(unpairGenP,axis=0)/(num_label+1)
        trip_loss = F.mean(F.maximum(zeros, margin+unpairSentProbs-pairSentProbs))
        return trip_loss
    
    vloss = triplet_loss(lm_flows['visF'], xp.array(num_labels['T']))
    lloss = triplet_loss(lm_flows['langF'], xp.array(num_labels['F']))
    #print(lang_loss, vloss, lloss)
    return langWeight*lang_loss + vlamda*vloss+llamda*lloss