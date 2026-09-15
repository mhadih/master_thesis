"""
This script provides an example to wrap UER-py for classification.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts, adv_opts
import pickle

import sys, os, difflib
import re
import codecs
import numpy
import numpy as np

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, \
    roc_auc_score, average_precision_score, precision_recall_curve


def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy == 1]) / len(testy)
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    return lr_auc


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        # self.labels_num = 1
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.max_pooler = torch.nn.AdaptiveMaxPool2d((1, args.hidden_size))
        self.hidden_state = args.hidden_size
        self.num_hunks = 5

    def forward(self, src, tgt, seg, soft_tgt=None):
            """
            Args:
                src: [batch_size x seq_length]
                tgt: [batch_size]
                seg: [batch_size x seq_length]
            """
            # Embedding.
            code_embed = []
            # print('tgt:', tgt.size())
            tgt = tgt[:, 0]

            for i in range(src.size()[1]):
                emb = self.embedding(src[:, i, :], seg[:, i, :])
                # Encoder.
                output = self.encoder(emb, seg[:, i, :])
                # Target.
                if self.pooling == "mean":
                    output = torch.mean(output, dim=1)
                elif self.pooling == "max":
                    output = torch.max(output, dim=1)[0]
                elif self.pooling == "last":
                    output = output[:, -1, :]
                else:
                    output = output[:, 0, :]
                output = torch.tanh(self.output_layer_1(output))
                # print('output', output.size())
                code_embed.append(output)

            code_embed = torch.cat(code_embed, 1)
            code_embed = code_embed.view(-1, 5, self.hidden_state)
            # print('code_embed: ',code_embed.size())
            max_code = self.max_pooler(code_embed)
            # print('x_code max pooling', max_code.size())
            x_code = max_code.squeeze(1)
            # print('x_code',x_code.size())

            logits = self.output_layer_2(x_code)
            # logits = self.output_layer_2(output)
            if tgt is not None:
                if self.soft_targets and soft_tgt is not None:
                    loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                           (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
                else:
                    loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
                return loss, logits
            else:
                return None, logits

    def count_labels_num(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        labels = []
        for i in range(len(data)):
            labels.append(data[i][0])
        labels_set = {}
        labels_set = set(labels)
        # print(labels_set)
        return len(labels_set)

    def load_or_initialize_parameters(args, model):
        if args.pretrained_model_path is not None:
            # Initialize with pretrained model.
            model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        else:
            # Initialize with normal distribution.
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    p.data.normal_(0, 0.02)

    def build_optimizer(args, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.optimizer in ["adamw"]:
            optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                      correct_bias=False)
        else:
            optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                      scale_parameter=False, relative_step=False)
        if args.scheduler in ["constant"]:
            scheduler = str2scheduler[args.scheduler](optimizer)
        elif args.scheduler in ["constant_with_warmup"]:
            scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
        else:
            scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
        return optimizer, scheduler

    def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
        instances_num = src.size()[0]

        for i in range(instances_num // batch_size):
            src_batch = src[i * batch_size: (i + 1) * batch_size, :]
            tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
            seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
            if soft_tgt is not None:
                soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
                yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
            else:
                yield src_batch, tgt_batch, seg_batch, None
        if instances_num > instances_num // batch_size * batch_size:
            src_batch = src[instances_num // batch_size * batch_size:, :]
            tgt_batch = tgt[instances_num // batch_size * batch_size:]
            seg_batch = seg[instances_num // batch_size * batch_size:, :]
            if soft_tgt is not None:
                soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
                yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
            else:
                yield src_batch, tgt_batch, seg_batch, None

    def read_dataset(args, path):
        with open(path, mode="rb") as f:
            data = pickle.load(f)
            dataset = []
            print('data size:', len(data))
            for i in range(len(data)):
                label, log, source_lists, target_lists = data[i]
                tgt = int(label)

                commit = []
                num_hunks = 5

                assert (len(source_lists) == len(target_lists))
                for j in range(len(source_lists)):
                    text_a, text_b = source_lists[j], target_lists[j]
                    text_a = args.tokenizer.convert_tokens_to_ids(text_a + ['</s>'])
                    text_b = args.tokenizer.convert_tokens_to_ids(text_b + ['</s>'])
                    text_log = args.tokenizer.convert_tokens_to_ids(['<s>'] + log + ['</s>'])
                    src = text_log + text_a + text_b
                    seg = [1] * len(text_log) + [1] * len(text_a) + [2] * len(text_b)
                    assert (len(src) == len(seg))

                    if len(src) > args.seq_length:
                        src = src[: args.seq_length]
                        seg = seg[: args.seq_length]

                    PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]

                    while len(src) < args.seq_length:
                        src.append(PAD_ID)
                        seg.append(0)
                    commit.append((src, tgt, seg))

                    if len(commit) > num_hunks:
                        commit = commit[:num_hunks]

                    while len(commit) < num_hunks:
                        src = [0] * args.seq_length
                        seg = [0] * args.seq_length
                        tgt = 0
                        commit.append((src, tgt, seg))

                dataset.append(commit)

        return dataset

    def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
        model.zero_grad()

        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        if soft_tgt_batch is not None:
            soft_tgt_batch = soft_tgt_batch.to(args.device)

        loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        if torch.cuda.device_count() > 1:
            loss = torch.mean(loss)

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if args.use_adv and args.adv_type == "fgm":
            args.adv_method.attack(epsilon=args.fgm_epsilon)
            loss_adv, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            if torch.cuda.device_count() > 1:
                loss_adv = torch.mean(loss_adv)
            loss_adv.backward()
            args.adv_method.restore()

        if args.use_adv and args.adv_type == "pgd":
            K = args.pgd_k
            args.adv_method.backup_grad()
            for t in range(K):
                # apply the perturbation to embedding
                args.adv_method.attack(epsilon=args.pgd_epsilon, alpha=args.pgd_alpha,
                                       is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    args.adv_method.restore_grad()
                loss_adv, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                if torch.cuda.device_count() > 1:
                    loss_adv = torch.mean(loss_adv)
                loss_adv.backward()
            args.adv_method.restore()

        optimizer.step()
        # scheduler.step()

        return loss

    def evaluate(args, dataset):

        src, tgt, seg = [], [], []
        for ii in range(len(dataset)):
            example = dataset[ii]
            src_hunk = [h[0] for h in example]
            tgt_hunk = [h[1] for h in example]
            seg_hunk = [h[2] for h in example]
            src.append(src_hunk)
            tgt.append(tgt_hunk)
            seg.append(seg_hunk)

        src = torch.LongTensor(src)
        tgt = torch.LongTensor(tgt)
        seg = torch.LongTensor(seg)
        # print(src.size())
        # print(seg.size())
        # print(tgt.size())

        batch_size = args.batch_size

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        args.model.eval()

        all_scores, all_golds = [], []
        for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            with torch.no_grad():
                _, logits = args.model(src_batch, tgt_batch, seg_batch)

            # print('tgt_batch:',tgt_batch.size())
            # print('src_batch:',src_batch.size())
            # print('logits:',logits.size())

            scores = nn.Softmax(dim=1)(logits)[:, 1]
            all_scores.extend(scores.detach().cpu().tolist())
            all_golds.extend(tgt_batch[:, 0].detach().cpu().tolist())

            pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            gold = tgt_batch[:, 0]

            # print('pred', pred, ' gold', gold)

            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()

        # print(all_scores, len(all_scores))
        # print(all_golds, len(all_golds))

        auc_score = roc_auc_score(y_true=all_golds, y_score=all_scores)
        pc = auc_pc(all_golds, all_scores)
        real_pred = [1 if p > 0.5 else 0 for p in all_scores]
        print('pred:', real_pred)
        print('labe:', all_golds)
        f1 = f1_score(y_true=all_golds, y_pred=real_pred)
        args.logger.info("AUC-ROC:{}  AUC-PR:{}  F1-Score:{}".format(auc_score, pc, f1))

        args.logger.debug("Confusion matrix:")
        args.logger.debug(confusion)
        args.logger.debug("Report precision, recall, and f1:")

        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 2 * p * r / (p + r + eps)
            args.logger.debug("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

        args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
        return correct / len(dataset), confusion

def main():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        finetune_opts(parser)

        parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="mean",
                            help="Pooling type.")

        tokenizer_opts(parser)

        parser.add_argument("--soft_targets", action='store_true',
                            help="Train model with logits.")
        parser.add_argument("--soft_alpha", type=float, default=0.5,
                            help="Weight of the soft targets loss.")

        adv_opts(parser)
        args = parser.parse_args()

        # Load the hyperparameters from the config file.
        args = load_hyperparam(args)
        # Count the number of labels.
        args.labels_num = count_labels_num(args.train_path)

        # Build tokenizer.
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        set_seed(args.seed)

        # Build classification model.
        model = Classifier(args)

        # Load or initialize parameters.
        load_or_initialize_parameters(args, model)

        # Get logger.
        args.logger = init_logger(args)

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(args.device)

        # Training phase.
        trainset = read_dataset(args, args.train_path)
        instances_num = len(trainset)
        batch_size = args.batch_size

        args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

        args.logger.info("Batch size: {}".format(batch_size))
        args.logger.info("The number of training instances: {}".format(instances_num))

        # optimizer, scheduler = build_optimizer(args, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            args.amp = amp

        if torch.cuda.device_count() > 1:
            args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        args.model = model

        if args.use_adv:
            args.adv_method = str2adv[args.adv_type](model)

        total_loss, result, best_result = 0.0, 0.0, 0.0

        args.logger.info("Start training.")
        for epoch in range(1, args.epochs_num + 1):
            random.shuffle(trainset)
            # print(type(trainset), len(trainset), type(trainset[0]), len(trainset[0]), type(trainset[0][0]), len(trainset[0][0]) , type(trainset[0][0][0]), len(trainset[0][0][0]), trainset[0][0][0], trainset[0][0][1], trainset[0][0][2] )
            print(np.shape(np.array(trainset)))
            print(np.shape(np.array(trainset[0])))
            src, tgt, seg = [], [], []
            for ii in range(len(trainset)):
                example = trainset[ii]
                src_hunk = [h[0] for h in example]
                tgt_hunk = [h[1] for h in example]
                seg_hunk = [h[2] for h in example]
                src.append(src_hunk)
                tgt.append(tgt_hunk)
                seg.append(seg_hunk)

            src = torch.LongTensor(src)
            tgt = torch.LongTensor(tgt)
            seg = torch.LongTensor(seg)
            print(src.size())
            print(seg.size())
            print(tgt.size())
            if args.soft_targets:
                soft_tgt = torch.FloatTensor([example[3] for example in trainset])
            else:
                soft_tgt = None

            model.train()
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(
                    batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                # print(src_batch)
                loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                if (i + 1) % args.report_steps == 0:
                    args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                                 total_loss / args.report_steps))
                    total_loss = 0.0

            result = evaluate(args, read_dataset(args, args.dev_path))
            if result[0] > best_result:
                best_result = result[0]
                save_model(model, args.output_model_path)

        if args.test_path is not None:
            args.logger.info("Test set evaluation.")
            if torch.cuda.device_count() > 1:
                args.model.module.load_state_dict(torch.load(args.output_model_path))
            else:
                args.model.load_state_dict(torch.load(args.output_model_path))
            evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()