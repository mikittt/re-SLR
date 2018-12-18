import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from models.base import LanguageEncoderAttn, LanguageEncoder, VisualEncoder, MetricNet
    
class ListenerReward(chainer.Chain):
    def __init__(self, vocab_size, attention=True, scale=1):
        if attention:
            le = LanguageEncoderAttn
        else:
            le = LanguageEncoder
            
        super(ListenerReward, self).__init__(
            ve = VisualEncoder(),
            le = le(vocab_size),
            me = MetricNet()
        )
        self.scale=scale
            
    def calc_score(self, feats, seq, lang_length):
        with chainer.using_config('train', False):
            vis_enc_feats = self.ve(feats)
            lang_enc_feats = self.le(seq, lang_length)
            lr_score = F.sigmoid(self.me(vis_enc_feats, lang_enc_feats))
        return lr_score
    
    def __call__(self, feats, seq, seq_prob, lang_length):#, baseline):
        xp = cuda.get_array_module(feats)
        
        lr_score = self.calc_score(feats, seq, lang_length).data[:,0]
        self.reward = lr_score*self.scale
        loss = -F.mean(F.sum(seq_prob, axis=0)/(xp.array(lang_length+1))*(self.reward-self.reward.mean()))
        return loss