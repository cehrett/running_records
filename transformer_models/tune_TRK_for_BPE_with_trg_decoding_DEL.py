import wandb
import transformer_repetition_kit as trk
import random
import numpy as np
import torch
from math import gcd
import time

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Use weights & biases to tune the transformer model.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data", help="data file path")
parser.add_argument("-m", "--max_evals", help="number of trials")
parser.add_argument('-s', "--sweep_id", help="Sweep ID for wandb")
args = parser.parse_args()
config = vars(args)

# Set number of evals that we can perform
max_evals = int(config['max_evals'])

# Set filepaths
ASR_df_filepath = config['data']
asr_text_filepath = 'asr.txt'
ttx_text_filepath = 'ttx.txt'
train_filename = 'train_sentence.csv'
valid_filename = 'valid_sentence.csv'
test_filename = 'test_sentence.csv'
trials_filename = f'{config["data"].split("/")[-1].split(".")[0]}.trials'

# Load data
trk.load_data(ASR_df_filepath=ASR_df_filepath,
              train_filename=train_filename,
              valid_filename=valid_filename,
              test_filename=test_filename,
              asr_text_filepath=asr_text_filepath,
              ttx_text_filepath=ttx_text_filepath)


def lcm(a, b):
    return abs(a*b) // gcd(a, b)


def main():
    """
    This is our main training routine, using weights and biases to track
    our hyperparamters, results, and suggest new hyperparemeters.
    """
    wandb.init()

    # Create tokenizer
    if wandb.config['bpe']:
        tokenizer = trk.create_train_bpe_tokenizer(wandb.config['bpe_vocab_size'],
                                                   asr_text_filepath=asr_text_filepath,
                                                   ttx_text_filepath=ttx_text_filepath,
                                                   save_tokenizer=False,
                                                   tokenizer_filename="./tokenizer-test.json"
                                                   )
    else:
        tokenizer = None

    # Preprocess data
    train_data, valid_data, test_data, TTX, TRG, ASR = trk.produce_iterators(train_filename,
                                                                             valid_filename,
                                                                             test_filename,
                                                                             asr_tokenizer=tokenizer,
                                                                             ttx_tokenizer=tokenizer
                                                                             )

    # Test out the tokenizer
    if wandb.config['bpe']:
        output = tokenizer.encode("Hello, y'all! How are you 😁 ? [WSP]")
        print(output.tokens)
        print(output.ids)

    # Tell Torch that we want to use the GPU
    device = torch.device('cuda')

    # Update params. This is to fet our hidden dimension number.
    wandb.config['hid_dim'] = lcm(
        wandb.config['enc_heads'], wandb.config['dec_heads']) * wandb.config['hid_dim_nheads_multiplier']

    # Train the model and get the loss
    model, train_loss, test_loss = trk.model_pipeline(device,
                                                      train_data,
                                                      valid_data,
                                                      test_data,
                                                      TTX,
                                                      TRG,
                                                      ASR
                                                      )

    # Log that loss to Weights & Biases as a Summary metric.
    wandb.run.summary['test_loss'] = test_loss

    torch.cuda.empty_cache()  # Needed so we don't kill off GPU Memory


if __name__ == '__main__':
    wandb.agent(args['sweep_id'], function=main, count=max_evals,
                project="running_records", entity="witw")
