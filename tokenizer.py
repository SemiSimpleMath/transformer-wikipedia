
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import config

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

def tokenizer_train():

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))


    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])


    tokenizer.pre_tokenizer = Whitespace()

    files = [f"./data/wikitext-103-raw-v1/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
    tokenizer.train(files, trainer)

    return tokenizer

def tokenizer_save(tok):
    tok.save("data/tokenizer-wiki-32k.json")

def tokenizer_load(path):
    tokenizer = Tokenizer.from_file(path)
    return tokenizer


def batch_iterator():
    bs = 1000
    for i in range(0, len(ds), bs):
        yield ds['train'][i : i + bs]["text"]

from datasets import load_dataset
ds = load_dataset("wikipedia", "20220301.en")


tok = Tokenizer(models.BPE())

tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(vocab_size=32767, special_tokens=["<|endoftext|>"])

tok.train_from_iterator(batch_iterator(), trainer=trainer)

tokenizer_save(tok)

encoding = tok.encode("Let's test this tokenizer.")
print(encoding.tokens)

tok.decoder = decoders.ByteLevel()

from transformers import GPT2TokenizerFast

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tok)

def load_tokenizer():
    toke = Tokenizer.from_file(config.tok_file)
    toke.decoder = decoders.ByteLevel()
    tok = GPT2TokenizerFast(tokenizer_object=toke)
    tok.add_special_tokens({'pad_token': '[PAD]'})

    return tok

