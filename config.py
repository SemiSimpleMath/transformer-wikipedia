# Model parameters
d_model = 512
h = 16
model_params = {'num_blocks': 16,
                'd_model': d_model,
                'd_middle': 4 * d_model,
                'dropout': 0.1,
                'h': 16,
                'd_q': d_model // h,
                'd_k': d_model // h,
                'd_v': d_model // h,
                }
# try with dropout = 0

seq_len = 256

model_directory = '.\\models\\'

batch_size = 16

num_batches =  500000
ds_path = "wikipedia"
ds_file = "20220301.en"
tok_file = "./data/wiki-tokens/tokenizer.json"

log_directory = "./logs/"
log_file = "log.txt"

save_every = 2000
output_every = 200