import math
import random

import torch
import torch.nn as nn
import decoder
import tokenizer
import utils
import time
# import tokenizer
import data_utils
import numpy as np
import datetime
import config
from datasets import load_dataset
import os
from tokenizers import Tokenizer, decoders
from transformers import GPT2TokenizerFast

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, opt, ds, tok, loss, bs, num_batches, seq_len, model_params):
    d_model = model_params['d_model']
    start_batch_num = model_params['start_batch_num']

    # params we track during training
    total_loss = 0
    total_sequences = start_batch_num * bs

    # lr schedule
    warmup_steps = 4000
    scheduled_lr = lambda b: d_model ** (-.5) * min((b + 1) ** (-.5), (b + 1) * warmup_steps ** (-1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=scheduled_lr)

    start_time = datetime.datetime.now()

    # mask and positional encoding
    mask = utils.get_mask(seq_len).to(device)
    pe = utils.get_pe(seq_len, d_model).to(device)

    for batch_num in range(start_batch_num, start_batch_num + num_batches):

        opt.zero_grad()

        # load batch
        combined = data_utils.get_justin_batch(ds, tok, bs, batch_num, seq_len + 1)
        src = combined[:, :-1].to(device)
        target = combined[:, 1:].to(device)

        # run through model
        pred = model(src, pe, mask)

        # compute loss
        pred = pred.permute(0, 2, 1)
        l = loss(pred, target)

        # back prop
        l.backward()

        # opt and scheduler step
        opt.step()
        scheduler.step()

        total_loss += l.item()
        total_sequences += bs * seq_len

        # log training data
        if batch_num % config.output_every == 0 and batch_num != start_batch_num:
            end_time = datetime.datetime.now()
            lr = opt.param_groups[0]["lr"]
            current_loss = total_loss / config.output_every
            total_loss = 0
            time_for_batch_interval = end_time - start_time

            log_data = {}
            file_id = model_params['id']
            log_data['batch_num'] = batch_num
            log_data['total_seq'] = total_sequences
            log_data['lr'] = lr
            log_data['current_loss'] = current_loss
            log_data['batch_interval'] = config.output_every
            log_data['time_for_batch_interval'] = time_for_batch_interval

            utils.log(file_id, log_data, log_screen=True)

            # pred = pred.permute(0, 2, 1)
            # print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')

            start_time = datetime.datetime.now()

        # save the model
        if batch_num % config.save_every == 0 and batch_num != start_batch_num:
            model_params['start_batch_num'] = batch_num + 1
            utils.save_model(model, opt, model_params)


def main():
    ds = load_dataset(config.ds_path, config.ds_file)
    tok = tokenizer.load_tokenizer()
    vocab_size = tok.vocab_size + 1
    ds = ds.shuffle(seed=42)

    LOAD = False
    if LOAD:
        directory = config.model_directory
        file = utils.most_recent_file(directory)
        model, opt, model_params = utils.load_model(file)
    else:
        model, opt, model_params = utils.default_model(vocab_size)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {pytorch_total_params}')

    utils.to_cuda(model)
    utils.optimizer_to(opt, device)

    loss = nn.CrossEntropyLoss()

    bs = config.batch_size
    num_batches = config.num_batches
    seq_len = config.seq_len
    train(model, opt, ds, tok, loss, bs, num_batches, seq_len, model_params)

    utils.save_model(model)


main()
