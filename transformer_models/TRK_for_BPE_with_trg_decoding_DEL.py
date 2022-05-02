# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import transformer_repetition_kit as trk
import random
import numpy as np
import torch


# %%
# Set config for this run
ASR_df_filepath = '../repetition_data_generation/data/output_del.csv'

config = dict(
    epochs=5,
    batch_size=128,
    learning_rate=0.01,
    dataset=ASR_df_filepath,
    hid_dim=256,
    enc_layers=4,
    dec_layers=4,
    enc_heads=8,
    dec_heads=8,
    enc_pf_dim=512,
    dec_pf_dim=512,
    enc_dropout=0.1,
    dec_dropout=0.2,
    clip=1,
    bpe_vocab_size=1600,
    decode_trg = True,
    early_stop = 6
)

asr_text_filepath = 'asr.txt'
ttx_text_filepath = 'ttx.txt'
train_filename = 'train_sentence.csv'
valid_filename = 'valid_sentence.csv'
test_filename = 'test_sentence.csv'


# %%
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# %%
#scr
import pandas as pd
df = pd.read_csv(ASR_df_filepath, names=['',
                                             'audio_path',
                                             'asr_transcript',
                                             'original_text',
                                             'mutated_text',
                                             'index_tags',
                                             'err_tags'], header=None, index_col='')


# %%
trk.load_data(ASR_df_filepath = ASR_df_filepath,
              train_filename = train_filename,
              valid_filename = valid_filename,
              test_filename = test_filename,
              asr_text_filepath = asr_text_filepath,
              ttx_text_filepath = ttx_text_filepath)


# %%
tokenizer = trk.create_train_bpe_tokenizer(config['bpe_vocab_size'],
                                           asr_text_filepath = \
                                           asr_text_filepath,
                                           ttx_text_filepath = ttx_text_filepath,
                                           save_tokenizer = True,
                                           tokenizer_filename = "./tokenizer-test.json"
                                          )


# %%
train_data, valid_data, test_data, TTX, TRG, ASR =     trk.produce_iterators(train_filename,
                          valid_filename,
                          test_filename,
                          asr_tokenizer=tokenizer,
                          ttx_tokenizer=tokenizer
                         )


# %%
# Test out the tokenizer
output = tokenizer.encode("Hello, y'all! How are you 😁 ? [WSP]")
print(output.tokens)
print(output.ids)


# %%
for i,t in enumerate(train_data): 
    if i<2: print(t.true_text,'\n',t.asr,'\n',t.tags,'\n')


# %%
torch.cuda.is_available()


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


# %%
model = trk.model_pipeline(config, 
                           device,
                           train_data,
                           valid_data,
                           test_data,
                           TTX,
                           TRG,
                           ASR
                          )


# %%



