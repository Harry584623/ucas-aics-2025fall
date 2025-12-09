# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import codecs
import os
import random
import time
import numpy as np

import torch
from hyperparams import Hyperparams as hp
from data_load import TestDataSet, load_de_vocab, load_en_vocab
from nltk.translate.bleu_score import corpus_bleu
from AttModel import AttModel
from torch.autograd import Variable

try:
    import cnmix
except ImportError:
    print("train without cnmix")


def _load_state_into_model(model, state):
    """兼容多种 checkpoint 保存格式。"""
    if isinstance(state, dict):
        if 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
            return
        if 'model' in state:
            model.load_state_dict(state['model'])
            return
    model.load_state_dict(state)


def eval(args):
    # Load data
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # TODO：加载德语/英语词表
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)

    # TODO：初始化模型
    model = AttModel(hp, enc_voc, dec_voc)
    print("AttModel PASS!")

    source_test = args.dataset_path + hp.source_test
    target_test = args.dataset_path + hp.target_test

    # TODO：测试集对象
    test_dataset = TestDataSet(source_test, target_test)

    # TODO：DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )

    if args.device == "MLU":
        model.mlu()
    elif args.device == "GPU":
        model.cuda()

    # TODO：加载权重
    state = torch.load(args.pretrained, map_location='cpu')
    # TODO：恢复权重
    _load_state_into_model(model, state)

    if args.device == "MLU" and args.cnmix:
        model, _ = cnmix.initialize(model, None, opt_level=args.opt_level)
        if isinstance(state, dict) and 'cnmix' in state:
            cnmix.load_state_dict(state['cnmix'])

    print('Model Loaded.')

    # TODO：eval 模式
    model.eval()

    # TODO：打开日志
    with codecs.open(args.log_path, 'a', 'utf-8') as fout:
        list_of_refs, hypotheses = [], []
        t1 = time.time()
        total_samples = 0

        # TODO：遍历测试集 (x, sources, targets)
        for i, (x, sources, targets) in enumerate(test_loader):
            if i == args.iterations:
                break

            total_samples += x.size(0)

            # Autoregressive inference
            if args.device == "GPU":
                x_ = x.long().cuda()
                preds_t = torch.LongTensor(np.zeros((x.size()[0], hp.maxlen), np.int32)).cuda()
                preds = Variable(preds_t).cuda()
            elif args.device == "MLU":
                x_ = x.long().to('mlu')
                preds_t = torch.LongTensor(np.zeros((x.size()[0], hp.maxlen), np.int32)).to('mlu')
                preds = Variable(preds_t.to('mlu'))
            else:
                x_ = x.long()
                preds_t = torch.LongTensor(np.zeros((x.size()[0], hp.maxlen), np.int32))
                preds = Variable(preds_t)

            for j in range(hp.maxlen):
                _, _preds, _ = model(x_, preds)
                preds_t[:, j] = _preds.data[:, j]
                preds = Variable(preds_t.long())

            preds = preds.data.cpu().numpy()

            # Write to file + accumulate refs/hypos
            for source, target, pred in zip(sources, targets, preds):
                got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                fout.write("- source: " + source + "\n")
                fout.write("- expected: " + target + "\n")
                fout.write("- got: " + got + "\n\n")
                fout.flush()

                ref = target.split()
                hypothesis = got.split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)

        # TODO：总时间
        temp_time = time.time() - t1
        print("time:", temp_time)

        # TODO：吞吐率 qps
        qps = (float(total_samples) / temp_time) if temp_time > 0 else 0.0
        print("qps:", qps)

        # TODO：BLEU
        score = corpus_bleu(list_of_refs, hypotheses)
        fout.write("Bleu Score = " + str(100 * score))
        print("Bleu Score = {}".format(100 * score))

    if os.getenv('AVG_LOG'):
        with open(os.getenv('AVG_LOG'), 'a') as train_avg:
            train_avg.write('Bleu Score:{}\n'.format(100 * score))

    print("Eval PASS!")


if __name__ == '__main__':
    # TODO：创建参数解析器
    parser = argparse.ArgumentParser(description="Transformer evaluation.")
    parser.add_argument('--device', default='MLU', type=str, help='set the type of hardware used for evaluation.')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--pretrained', default='model_epoch_20.pth', type=str, help='training ckps path')
    parser.add_argument('--batch-size', default=32, type=int, help='evaluation batch size.')
    parser.add_argument('--workers', default=4, type=int, help='number of workers.')
    parser.add_argument('--log-path', default='output.txt', type=str, help='evaluation file path.')
    parser.add_argument('--dataset-path', default='corpora/', type=str, help='The path of dataset.')
    parser.add_argument('--iterations', default=-1, type=int, help="Number of eval iterations (-1 means all).")
    parser.add_argument('--bitwidth', default=8, type=int, help="Set the initial quantization width of network training.")
    parser.add_argument('--cnmix', action='store_true', default=False, help='use cnmix for mixed precision training')
    parser.add_argument('--opt_level', type=str, default="O0", help='choose level of mixing precision')

    # TODO：解析参数
    args = parser.parse_args()

    if args.device == "MLU":
        import torch_mlu  # noqa: F401

    # TODO：开始评估
    eval(args)

    if args.device == "MLU":
        print("Transformer MLU PASS!")
    else:
        print("Transformer CPU PASS!")
