import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    # output json
    parser.add_argument('--data_json', default='data.json', help='output json file')
    parser.add_argument('--data_h5', default='data.h5', help='output h5 file')
    parser.add_argument('--image_feats', default='image_feats.npy', help='image, Variable feats file')
    parser.add_argument('--ann_feats', default='ann_feats.npy', help='ann feats file')
    parser.add_argument('--save_dir', default='./save_dir')
    parser.add_argument('--old', '-old', action='store_true')

    # options
    parser.add_argument('--data_root', default='./dataset/anns/original', type=str, help='data folder containing images and four datasets.')
    parser.add_argument('--dataset', '-d', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
    parser.add_argument('--splitBy', '-s', default='unc', type=str, help='unc/google')
    parser.add_argument('--max_length', type=int, help='max length of a caption')  # refcoco 10, refclef 10, refcocog 20, refgta 20
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

    parser.add_argument('--mscoco_root', default='./dataset/coco_image/train2014', type=str)
    parser.add_argument('--gta_root', default='./dataset/gta_image/gtav_cv_mod_data', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu_id', '-g', type=int, default=0)
    
    parser.add_argument('--word_emb_path', default='word_emb.npy')
    parser.add_argument('--sample_ratio', type=float, default=0.5)
    parser.add_argument('--sample_neg', type=int, default=1)
    parser.add_argument('--hard_temperature', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--seq_per_ref', type=int, default=3)
    parser.add_argument('--mine_hard_every', type=int, default=4000)
    parser.add_argument('--learning_rate_decay_start', type=int, default=8000)
    parser.add_argument('--learning_rate_decay_every', type=int, default=8000)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    parser.add_argument('--losses_log_every', type=int, default=25)
    parser.add_argument('--max_iter', type=int, default=-1)
    parser.add_argument('--save_checkpoint_every', type=int, default=2000)
    # language encoder
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--optim_alpha', type=float, default=0.8)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    #visual encoder
    parser.add_argument('--ve_learning_rate', type=float, default=4e-5)
    parser.add_argument('--ve_optim_alpha', type=float, default=0.8)
    parser.add_argument('--ve_optim_beta', type=float, default=0.999)
    
    parser.add_argument('--id', '-id',default='met')
    parser.add_argument('--id2', '-id2',default='')
    parser.add_argument('--pretrained_w', '-pw', action='store_true')
    
    parser.add_argument('--generation_weight', type=float, default=1)
    parser.add_argument('--vis_rank_weight', type=float, default=1)
    parser.add_argument('--lang_rank_weight', type=float, default=0)
    parser.add_argument('--embedding_weight', type=float, default=1)
    parser.add_argument('--lm_margin', type=float, default=1)
    parser.add_argument('--emb_margin', type=float, default=0.1)
    
    parser.add_argument('--check_sent', '-check', action='store_true')
    parser.add_argument('--mine_hard', '-mine', action='store_true')
    
    parser.add_argument('--split', '-split',default='testA')
    parser.add_argument('--mode','-mode',type=int, default=1)
    parser.add_argument('--lamda','-lam',type=float, default=0.2)
    parser.add_argument('--beam_width','-beam',type=int, default=10)
    parser.add_argument('--write_result', default=0, type=int)
    
    # argparse
    args = parser.parse_args()
    
    return args