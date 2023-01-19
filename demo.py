from tokenizers import Tokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

import decoder
import utils
import time

import tokenizer
import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def demo_model(model, DL, seq_len, d_model,  input=None, babble_len = 0):
    model.eval()
    pe = decoder.get_pe(seq_len, d_model)
    msk = decoder.get_mask(seq_len)
    if input is None:
        bs = 2
        src, target = DL.get_batch(bs)
        src = src.to(device)
        target = target.to(device)



        pred = model(src, pe, msk)
        pred = F.softmax(pred,-1)
        pred = torch.argmax(pred, dim=-1)


def babble(model, prompt, d_model, tok, N):
    src = prompt.unsqueeze(0)


    for _ in range(N):
        seq_len = src.shape[-1]
        pe = decoder.get_pe(seq_len, d_model)
        pe = pe.to(device)
        msk = decoder.get_mask(seq_len)
        msk = msk.to(device)

        #shifted = src[0,1:]
        shifted = src
        pred = model(src, pe, msk)
        pred = F.softmax(pred,-1)
        pred = torch.argmax(pred, dim=-1)
        last = pred[0][-1]

        last = last.unsqueeze(0).unsqueeze(0)
        shifted = torch.cat([shifted, last], -1)


        src = shifted


    t = data_utils.tensor_to_token_ids(shifted)
    print(t)
    #text = tok.decode_batch([t])
    text = data_utils.tokens_to_words(t, tok)
    print(text)

def main():

    tokenizer_path = "./data/wikitext-103-raw/tokenizer-wiki.json"
    tok = tokenizer.tokenizer_load(tokenizer_path)


    # Model paramaters
    num_blocks = 8
    d_model = 512
    d_middle = 4 * d_model
    vocab_size = tok.get_vocab_size() + 1
    dropout = 0.1
    h = 8
    d_Q = d_model//h
    d_K = d_model//h
    d_V = d_model//h
    MAX_SEQ_LEN = 200
    pos_enc = decoder.positional_encoding(MAX_SEQ_LEN, d_model)


    pos_enc = pos_enc.to(device)
    mask = decoder.create_upper_mask(MAX_SEQ_LEN + 2)
    mask = mask.to(device)



    seq_len = 100



    model = decoder.Decoder(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_Q, d_K, d_V)

    LOAD = True
    if LOAD:
        PATH = '.\models\model-2-20230117-160535'
        model.load_state_dict(torch.load(PATH))


    utils.to_cuda(model)

    model.eval()


    prompt = data_utils.text_to_model_input("Once upon a time there lived a little country girl, the prettiest creature who was ever seen. Her mother had a little red riding hood made for her. Everybody called her Little Red Riding Hood.", tok)
    prompt = prompt.to(device)
    seq_len = prompt.shape[-1]
    babble_len = 100
    babble(model, prompt, d_model, tok, babble_len)

main()


