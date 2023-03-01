import wandb
import transformer_repetition_kit as trk
import torch
from math import gcd
import traceback
import sys
import argparse
from tempfile import NamedTemporaryFile
from pathlib import Path

SCRATCH_DIR = Path('/scratch1/jmdanie')

# Parse command line arguments
parser = argparse.ArgumentParser(description="Use weights & biases to tune the transformer model.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data", help="data file path", required=True)
parser.add_argument("-m", "--max_evals", help="number of trials", required=True)
parser.add_argument('-s', "--sweep_id", help="Sweep ID for wandb", required=True)
parser.add_argument('-t', "--error_tag", help="The tag we are interested in tracking. (e.g. -, S, REP)", required=True, nargs='+')
args = parser.parse_args()
config = vars(args)

# Set number of evals that we can perform
max_evals = int(config['max_evals'])

def lcm(a, b):
    return abs(a*b) // gcd(a, b)


def main():
    """
    This is our main training routine, using weights and biases to track
    our hyperparamters, results, and suggest new hyperparemeters.
    """
    try:
        wandb.init(dir=SCRATCH_DIR.joinpath('wandb'))
        wandb.config['data'] = config['data']
        wandb.config['error_tag'] = config['error_tag']

        # Set filepaths. We will create temporary files to store the data. This also allows
        # us to train on different hosts.
        temp_dir = SCRATCH_DIR.joinpath('temp_data')

        ASR_df_filepath = config['data']
        asr_text_file = NamedTemporaryFile(mode='w', prefix='asr', suffix='.txt', delete=True, dir=temp_dir)
        ttx_text_file = NamedTemporaryFile(mode='w', prefix='ttx', suffix='.txt', delete=True, dir=temp_dir)
        train_file = NamedTemporaryFile(mode='w', prefix='train_sentence', suffix='.csv', delete=True, dir=temp_dir)
        valid_file = NamedTemporaryFile(mode='w', prefix='valid_sentence', suffix='.csv', delete=True, dir=temp_dir)
        test_file = NamedTemporaryFile(mode='w', prefix='test_sentence', suffix='.csv', delete=True, dir=temp_dir)

        # Load data
        trk.load_data(ASR_df_filepath=ASR_df_filepath,
                    train_filename=train_file.name,
                    valid_filename=valid_file.name,
                    test_filename=test_file.name,
                    asr_text_filepath=asr_text_file.name,
                    ttx_text_filepath=ttx_text_file.name)

        # Create tokenizer
        if wandb.config['bpe']:
            tokenizer = trk.create_train_bpe_tokenizer(wandb.config['bpe_vocab_size'],
                                                       asr_text_filepath=asr_text_file.name,
                                                       ttx_text_filepath=ttx_text_file.name,
                                                       save_tokenizer=False,
                                                       tokenizer_filename="./tokenizer-test.json"
                                                       )
        else:
            tokenizer = None

        # Preprocess data
        train_data, valid_data, test_data, TTX, TRG, ASR, TTX_POS, ASR_POS = trk.produce_iterators(train_file.name,
                                                                                                   valid_file.name,
                                                                                                   test_file.name,
                                                                                                   asr_tokenizer=tokenizer,
                                                                                                   ttx_tokenizer=tokenizer
                                                                                                   )

        # Close the temporary files I created earlier
        train_file.close()
        valid_file.close()
        test_file.close()
        asr_text_file.close()
        ttx_text_file.close()

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
                                                          ASR,
                                                          TTX_POS,
                                                          ASR_POS,
                                                          config['error_tag']
                                                          )

        # Log that loss to Weights & Biases as a Summary metric.
        wandb.run.summary['test_loss'] = test_loss
        wandb.run.summary['train_loss'] = train_loss

        torch.cuda.empty_cache()  # Needed so we don't kill off GPU Memory
    except Exception as e:
        print(e)
        print(traceback.print_exc(), file=sys.stderr)
        raise e


if __name__ == '__main__':
    wandb.agent(config['sweep_id'], function=main, count=max_evals,
                project="running_records", entity="witw")
