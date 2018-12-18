import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda

class CcaEmbedding(chainer.Chain):
    def __init__(self, vis_enc_size=512, lang_enc_size=512, emb_size=512):
        initializer = chainer.initializers.GlorotNormal(scale=math.sqrt(2))
        super(CcaEmbedding, self).__init__(
            normv0 = L.BatchNormalization(512, eps=1e-5),
            v1 = L.Linear(vis_enc_size, emb_size, initialW=initializer),
            normv1 = L.BatchNormalization(emb_size, eps=1e-5),
            v2 = L.Linear(emb_size, emb_size, initialW=initializer),
            normv2 = L.BatchNormalization(emb_size, eps=1e-5),
            
            l1 = L.Linear(lang_enc_size, emb_size, initialW=initializer),
            norml1 = L.BatchNormalization(emb_size, eps=1e-5),
            l2 = L.Linear(emb_size, emb_size, initialW=initializer),
            norml2 = L.BatchNormalization(emb_size, eps=1e-5),
        )
        
    def vis_forward(self, vis):
        vis1 = F.dropout(F.relu(self.normv1(self.v1(self.normv0(vis)))), ratio=0.1)
        vis2 = F.normalize(self.normv2(self.v2(vis1)))
        return vis2
    
    def lang_forward(self, lang):
        lang1 = F.dropout(F.relu(self.norml1(self.l1(lang))), ratio=0.1)
        lang2 = F.normalize(self.norml2(self.l2(lang1)))
        return lang2
    
    def __call__(self, vis, lang):
        vis2 = self.vis_forward(vis)
        lang2 = self.lang_forward(lang)
        dot_product = F.sum(vis2*lang2, axis=1)
        return dot_product, vis2