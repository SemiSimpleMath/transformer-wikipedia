import torch
import pandas as pd
import re
import random


class DataLoader:
    def __init__(self, data, tok):
        self.ds = data
        self.tok = tok

    def get_batch(self, bs, min_seq_len):
        src_batch = []
        target_batch = []


        while(len(src_batch) < bs):
            r = random.randint(0, 6458670-1)
            article = self.ds['train'][r]['text']
            article = article.split()


            while len(article) < min_seq_len + 1:
                r = random.randint(0, 6458670-1)
                article = self.ds['train'][r]['text']
                article = article.split()

            art_len = len(article)
            start = random.randint(0, art_len - min_seq_len - 1)
            start = max(0, start)
            src = article[start:start+min_seq_len]
            target = article[start+1:start+min_seq_len+1]

            src = self.tok.encode(' '.join(src))
            src = torch.LongTensor(src.ids)
            target = self.tok.encode(' '.join(target))
            target = torch.LongTensor(target.ids)

            src = src[0:min_seq_len]
            target = target[0:min_seq_len]

            src_batch.append(src.unsqueeze(0))
            target_batch.append(target.unsqueeze(0))


        src_batch = torch.cat(src_batch, dim=0)
        target_batch = torch.cat(target_batch, dim=0)
        return src_batch, target_batch


def text_to_token(text, tokenizer):
    output = tokenizer.encode(text)
    return output.ids

def tokens_to_tensor(tokens):
    return torch.LongTensor(tokens.ids)

def tensor_to_token_ids(t):
    t = t.squeeze()
    t = t.tolist()
    return t

def token_id_to_word(t, tokenizer):
    return tokenizer.id_to_token(t)

def tokens_to_words(t, tokenizer):
    result = ""
    for id in t:
        result += " " + token_id_to_word(id, tokenizer)
    return result

def text_to_model_input(text, tokenizer):
    tokens = text_to_token(text, tokenizer)
    return torch.LongTensor(tokens)