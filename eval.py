# coding=utf-8
import os
from evaluation import eval_DECL

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # DECL-SGRAF. Ensemble the SGR and SAF models
    avg_SGRAF = True
    
    data_path= 'the datasets path'
    vocab_path= 'the vocab path/'
    ns = [0.2, 0.5, 0.7]
    for n in ns:
        print(f"\n============================>f30k noise:{n}")
        checkpoint_paths = [
            f'./f30k_SAF_noise{n}_best.tar',
            f'./f30k_SGR_noise{n}_best.tar']
        eval_DECL(checkpoint_paths, avg_SGRAF, data_path=data_path, vocab_path=vocab_path)

    for n in ns:
        print(f"\n============================>coco noise:{n}")
        checkpoint_paths = [
            f'./coco_SAF_noise{n}_best.tar',
            f'./coco_SGR_noise{n}_best.tar']
        eval_DECL(checkpoint_paths, avg_SGRAF, data_path=data_path, vocab_path=vocab_path)

    print(f"\n============================>cc152k")
    eval_DECL(['./cc152k_SAF_best.tar', './cc152k_SGR_best.tar'], avg_SGRAF=avg_SGRAF)
