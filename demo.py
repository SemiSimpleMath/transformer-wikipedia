
import torch
import torch.nn.functional as F
import decoder
import config
import numpy as np
import tokenizer
import data_utils
import math
from torch.autograd import Variable
import utils

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
    return Variable(msk, requires_grad=False)


MAX_SEQ_LEN = 500
MAX_D_MODEL = 512

pos_enc = positional_encoding(MAX_SEQ_LEN, MAX_D_MODEL)
pos_enc = pos_enc.to(device)


def get_pe(seq_len, d_model):
    pe = pos_enc[0, :seq_len, :d_model]
    return Variable(pe, requires_grad=False)


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

    # mask and positional encoding



    src = prompt.unsqueeze(0)


    for _ in range(N):
        seq_len = src.shape[-1]

        mask = utils.get_mask(seq_len).to(device)
        pe = utils.get_pe(seq_len, d_model).to(device)

        shifted = src
        pred = model(src, pe, mask)
        pred = F.softmax(pred,-1)
        pred = torch.argmax(pred, dim=-1)
        last = pred[0][-1]

        last = last.unsqueeze(0).unsqueeze(0)
        shifted = torch.cat([shifted, last], -1)


        src = shifted


    t = data_utils.tensor_to_token_ids(shifted)
    print(t)

    text = data_utils.tokens_to_words(t, tok)
    print(text)



def main():

    tok = tokenizer.load_tokenizer()
    vocab_size = tok.vocab_size + 1

    directory = config.model_directory
    file = utils.most_recent_file(directory)
    model, opt, model_params = utils.load_model(file)


    utils.to_cuda(model)

    model.eval()

    prompt = data_utils.text_to_model_input("Rahul Fernandez is a mathematician who", tok)
    prompt = prompt.to(device)
    seq_len = prompt.shape[-1]
    babble_len = 100
    babble(model, prompt, model_params, tok, babble_len)

main()


def nucleus_sampling(b, p):
    probs = F.softmax(b, dim=-1)
    log_probs = torch.log(probs)
    sorted_log_probs, indices = torch.sort(log_probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_log_probs, dim=-1)
    nucleus = cum_sum_probs < p
    sorted_log_probs[~nucleus] = float('-inf')
    next_token_indices = torch.zeros_like(probs)
    for i in range(b.shape[0]):
        next_token_indices[i] = torch.multinomial(probs[i], 1)
    return next_token_indices


def beam_search():
    # for each batch
    # get top k results

    probs, idx = d.topk(3)



    # determine which sentences will go on to the next round

    # create the new batches

    # repeat for desired number of steps

    # pick the sentence with highest score.



    return




