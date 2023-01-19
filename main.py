import math
import torch
import torch.nn as nn
import decoder
import utils
import time
import tokenizer
import data_utils
import numpy as np

from torch.autograd import Variable

from datasets import load_dataset

ds = load_dataset("wikipedia", "20220301.en")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_by_position(pred, target, bs, seq_len, loss):
    loss_by_pos = [0] * seq_len
    for row1, row2 in zip(pred, target):
        pos = 0
        for seq, target in zip(row1, row2):
            loss_by_pos[pos] += (loss(seq, target).item())
            pos += 1

    loss_by_pos[:] = [x / bs for x in loss_by_pos]

    return loss_by_pos


def save_model(model, opt, batch_num=0):
    path = '.\models\model-5-' + time.strftime("%Y%m%d-%H%M%S")
    model.state_dict()['batch_num'] = batch_num

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'batch_num': batch_num,
    }, path)

    print(f"Saving model: {path}")


def train(model, opt, loss, bs, num_batches, d_model, DL, seq_len, start_batch_num=0):
    total_loss = 0
    output_every = 500
    total_sequences = 0

    warmup_steps = 4000

    lambda1 = lambda b: d_model ** (-.5) * min((b + 1) ** (-.5), (b + 1) * warmup_steps ** (-1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
    lrs = []
    for batch_num in range(start_batch_num, start_batch_num + num_batches):
        opt.zero_grad()
        src, target = DL.get_batch(bs, seq_len)

        src = src.to(device)
        target = target.to(device)

        pe = decoder.get_pe(seq_len, d_model)
        msk = decoder.get_mask(seq_len)
        pe = pe.unsqueeze(0)
        msk = msk.unsqueeze(0)

        pred = model(src, pe, msk)
        pred = pred.permute(0, 2, 1)

        total_sequences += bs * seq_len

        # Compute the loss
        l = loss(pred, target)
        total_loss += l.item()

        # Backward pass
        l.backward()

        # Update the parameters
        opt.step()
        scheduler.step()

        if batch_num % output_every == 0 and batch_num != 0:
            print(total_loss / output_every)
            print(l)
            total_loss = 0
            print(f'Batch number: {batch_num}')
            print("total sequences: ", total_sequences)
            lrs.append(opt.param_groups[0]["lr"])
            print("Current lr: ", lrs[-1])
            pred = pred.permute(0, 2, 1)
            print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')

        if batch_num % 2000 == 0 and batch_num != 0:
            save_model(model, opt, batch_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_upper_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    # noinspection PyUnresolvedReferences
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        create_upper_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe



MAX_MASK_SIZE = 512
mask = create_upper_mask(MAX_MASK_SIZE)
mask = mask.to(device)


def get_mask(size):
    msk = mask[0, :size, : size]
    return msk


MAX_SEQ_LEN = 500
MAX_D_MODEL = 512

pos_enc = positional_encoding(MAX_SEQ_LEN, MAX_D_MODEL)
pos_enc = pos_enc.to(device)


def get_pe(seq_len, d_model):
    pe = pos_enc[0, :seq_len, :d_model]
    return Variable(pe, requires_grad=False)


def main():
    tokenizer_path = "./data/wikitext-103-raw/tokenizer-wiki.json"
    tok = tokenizer.tokenizer_load(tokenizer_path)

    # Model paramaters
    num_blocks = 6
    d_model = 512
    d_middle = 4 * d_model
    vocab_size = tok.get_vocab_size()
    dropout = 0.1
    h = 8
    d_Q = d_model // h
    d_K = d_model // h
    d_V = d_model // h

    seq_len = 100

    DL = data_utils.DataLoader(ds, tok)

    model = decoder.Decoder(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_Q, d_K, d_V)
    opt = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    start_batch_num = 0
    LOAD = False
    if LOAD:
        path = '.\\models\\'
        file = 'model-4-20230118-183617'

        checkpoint = torch.load(path + file)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_batch_num = checkpoint['batch_num']
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(pytorch_total_params)
    utils.to_cuda(model)
    utils.optimizer_to(opt, device)

    loss = nn.CrossEntropyLoss()

    bs = 32
    num_batches = 500000
    train(model, opt, loss, bs, num_batches, d_model, DL, seq_len, start_batch_num)

    save_model(model)


# token 3 is [PAD]

# there are 6458670 articles

main()
