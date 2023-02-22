"""
This code is adapted from the tutorial found [here](https://github.com/bentrevett/pytorch-seq2seq), on implementing the
Transformer from "[Attention is All You Need](https://arxiv.org/abs/1706.03762)".
"""

import torch
import torch.nn as nn
from torch import Tensor

from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from string import punctuation

import math
import time
import wandb

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import numpy as np
import traceback

from typing import List, Union

import socket

# This enables us to be running multiple jobs in parallel and not write to the same
# files. This helps us avoid an error where we try to avoid loading the wrong models.
MODEL_SAVE_FILENAME = socket.gethostname().split('.')[0] + '_model.pt'


def load_data(ASR_df_filepath='/home/cehrett/running_records/repetition_data_generation/data/generated_data.csv',
              train_filename='train_sentence.csv',
              valid_filename='valid_sentence.csv',
              test_filename='test_sentence.csv',
              asr_text_filepath='asr.txt',
              ttx_text_filepath='ttx.txt'):
    """
    Load the data and create train/val/test csvs, and also create .txt files of true text and ASR for BPE.
    :param ASR_df_filepath: Location of data
    :param train_filename: filename to which to write train data
    :param valid_filename: filename to which to write valid data
    :param test_filename: filename to which to write test data
    :param asr_text_filepath: filename to which to write asr transcripts
    :param ttx_text_filepath: filename to which to write true text transcripts
    :return:
    """
    # Get data
    df = pd.read_csv(ASR_df_filepath, names=['audio_path',
                                             'asr_transcript',
                                             'original_text',
                                             'mutated_text',
                                             'index_tags',
                                             'err_tags'], header=0)

    # Lowercase all true text
    df.loc[:, 'original_text'] = df.original_text.str.lower()

    # Add duplicate columns to facilitate the acquisition of pos embeddings
    df['original_text_pos'] = df['original_text']
    df['asr_transcript_pos'] = df['asr_transcript']

    # Data is already shuffled, so no need to do that now.
    # TODO verify this is still the case; probably not

    df_len = df.shape[0]

    # Make train/val/test split
    # TODO make configurable
    train_len = round(df_len * .6)
    val_len = round(df_len * .2)

    df_train = df[:train_len]
    df_val = df[train_len:train_len + val_len]
    df_test = df[train_len + val_len:]

    df_train.to_csv(train_filename)
    df_val.to_csv(valid_filename)
    df_test.to_csv(test_filename)

    # Save the sample TTX and ASR output by itself to a text file
    # Once dataset is big enough, this should use only training data. Now it uses all
    df_asr = df[['asr_transcript']]
    df_ttx = df[['original_text']]
    df_asr.to_csv(asr_text_filepath, sep='\t', header=False, index=False)
    df_ttx.to_csv(ttx_text_filepath, sep='\t', header=False, index=False)


def create_train_bpe_tokenizer(bpe_vocab_size,
                               asr_text_filepath='asr.txt',
                               ttx_text_filepath='ttx.txt',
                               save_tokenizer=True,
                               tokenizer_filename="./data/tokenizer-test.json"):
    """
    TODO produce documentation.

    :param bpe_vocab_size:
    :param asr_text_filepath:
    :param ttx_text_filepath:
    :param save_tokenizer:
    :param tokenizer_filename:
    :return:
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                         vocab_size=bpe_vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    files = [asr_text_filepath, ttx_text_filepath]
    files = [file for file in files if file]  # Get rid of None's
    tokenizer.train(files, trainer)

    if save_tokenizer:
        tokenizer.save(tokenizer_filename)

    return tokenizer


def tokenize_TTX(text, tokenizer=None):
    """
    Tokenizes English text from a string into a list of strings
    """
    if tokenizer:
        return tokenizer.encode(text).tokens
    else:
        return [tok.strip(punctuation) for tok in text.split(" ") if tok not in ['.', ',', '!', '?', ';', ':', ]]


def tokenize_TRG(tags):
    """
    Tokenizes string representation of array into list of strings
    """
    tags = tags[1:-1]
    return [tag.strip(" '") for tag in tags.split(',')]


def tokenize_ASR(asr, tokenizer):
    """
    Tokenizes ASR transcript from a string into a list of strings
    """
    if tokenizer:
        return tokenizer.encode(asr).tokens
    else:
        return [tok.strip(punctuation) for tok in asr.split(" ") if tok not in ['.', ',', '!', '?', ';', ':', ]]


def pos_embed(text, tokenizer=None):
    """
    Provides position info about English text
    """
    if tokenizer:
        return tokenizer.encode(text).word_ids
    else:
        return [i for i, tok in enumerate(text.split(" ")) if tok not in ['.', ',', '!', '?', ';', ':', ]]


def produce_iterators(train_filename,
                      valid_filename,
                      test_filename,
                      asr_tokenizer=None,
                      ttx_tokenizer=None
                      ):
    """
    Produce datasets for each of training, validation and test data. Also build vocabs for true text, tags, and ASR.
    :param train_filename: location of train data csv
    :param valid_filename: location of valid data csv
    :param test_filename: location of test data csv
    :return:
    """
    TTX = Field(tokenize=lambda x: tokenize_TTX(x, ttx_tokenizer),
                init_token='<sos>',
                eos_token='<eos>',
                lower=False,
                batch_first=True)

    TRG = Field(tokenize=tokenize_TRG,
                init_token='<sos>',
                eos_token='<eos>',
                lower=False,
                batch_first=True)

    ASR = Field(tokenize=lambda x: tokenize_ASR(x, asr_tokenizer),
                init_token='<sos>',
                eos_token='<eos>',
                lower=False,
                batch_first=True)

    TTX_POS = Field(tokenize=lambda x: pos_embed(x, ttx_tokenizer),
                    use_vocab=False,
                    init_token=0,
                    eos_token=0,
                    lower=False,
                    batch_first=True,
                    pad_token=0)

    ASR_POS = Field(tokenize=lambda x: pos_embed(x, asr_tokenizer),
                    use_vocab=False,
                    init_token=0,
                    eos_token=0,
                    lower=False,
                    batch_first=True,
                    pad_token=0)

    fields = {'original_text': ('true_text', TTX),
              'err_tags': ('tags', TRG),
              'asr_transcript': ('asr', ASR),
              'original_text_pos': ('true_text_pos', TTX_POS),
              'asr_transcript_pos': ('asr_pos', ASR_POS)}

    train_data, valid_data, test_data = TabularDataset.splits(
        path='./',
        train=train_filename,
        validation=valid_filename,
        test=test_filename,
        format='csv',
        fields=fields
    )

    # Put min_freq at 2 or higher for real data
    TTX.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    ASR.build_vocab(train_data, min_freq=2)

    # Return datasets along with vocab objects for each of TTX,TRG,ASR
    return train_data, valid_data, test_data, TTX, TRG, ASR, TTX_POS, ASR_POS


def model_pipeline(device,
                   train_data,
                   valid_data,
                   test_data,
                   TTX,
                   TRG,
                   ASR,
                   TTX_POS,
                   ASR_POS,
                   error_tags
                   ):
    # access all HPs through wandb.config, so logging matches execution!
    config = wandb.config

    # make the model, data, and optimization problem
    model, train_iterator, valid_iterator, test_iterator, criterion, optimizer = make(config,
                                                                                      device,
                                                                                      train_data,
                                                                                      valid_data,
                                                                                      test_data,
                                                                                      TTX,
                                                                                      TRG,
                                                                                      ASR
                                                                                      )
    #       print(model)

    # and use them to train the model
    model, train_loss = train(
        model, train_iterator, valid_iterator, criterion, optimizer, config, TTX, TRG, ASR, TTX_POS, ASR_POS, error_tags)

    # and test its final performance
    model, test_loss = test(model, test_iterator, criterion,
                            TTX, TRG, ASR, error_tags)

    return model, train_loss, test_loss


def make(config,
         device,
         train_data,
         valid_data,
         test_data,
         TTX,
         TRG,
         ASR):
    # Make the data
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key=lambda x: len(x.true_text),
        batch_size=config.batch_size,
        device=device)

    # Make the model
    model = make_model(config, device, TTX, TRG, ASR)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    wandb.summary['trainable_parameters'] = count_parameters(model)
    model.apply(initialize_weights)

    # Make the loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config['decode_trg']: 
        ignore_index = TRG.vocab.stoi[TRG.pad_token]
    else:
        ignore_index = TTX.vocab.stoi[TTX.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    return model, train_iterator, valid_iterator, test_iterator, criterion, optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def make_model(config, device, TTX, TRG, ASR):
    TTX_INPUT_DIM = len(TTX.vocab)
    ASR_INPUT_DIM = len(ASR.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = config.hid_dim
    ENC_LAYERS = config.enc_layers
    DEC_LAYERS = config.dec_layers
    ENC_HEADS = config.enc_heads
    DEC_HEADS = config.dec_heads
    ENC_PF_DIM = config.enc_pf_dim
    DEC_PF_DIM = config.dec_pf_dim
    ENC_DROPOUT = config.enc_dropout
    DEC_DROPOUT = config.dec_dropout

    ttx_enc = Encoder(TTX_INPUT_DIM,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      device)

    asr_enc = Encoder(ASR_INPUT_DIM,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      device)

    if config['decode_trg'] == True:
        Decoder = Decoder_trg
        Seq2Seq = Seq2Seq_trg
    else:
        Decoder = Decoder_ttx
        Seq2Seq = Seq2Seq_ttx
    dec = Decoder(TTX_INPUT_DIM,
                  OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device)

    TTX_PAD_IDX = TTX.vocab.stoi[TTX.pad_token]
    ASR_PAD_IDX = ASR.vocab.stoi[ASR.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(ttx_enc, asr_enc, dec, TTX_PAD_IDX,
                    ASR_PAD_IDX, TRG_PAD_IDX, device).to(device)
    return model

def get_positives_and_negatives(output: torch.Tensor, trg: torch.Tensor, target_label: int, pad_label: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    This will report the number of true positives, false positives, and false negatives
    given a target_label and a pad_label. The target_label is the label we are
    interested in getting the stats for, and the pad_label is the one that represents
    the PAD token in the output. 
    """

    # output should be the softamx outputs of the model, and trg
    # should be the true labels. 
    cur_output = output.clone().cpu()
    cur_trg = trg.clone().cpu()

    # Get the predicted class by taking the argmax of each word
    # The model will return a list of predictions for each input token.
    cur_output = cur_output.argmax(dim=1)

    # Remove all indexes where the correct label is a PAD token. We don't care
    # about these in our calculaton. Both cur_output and trg need to have the same
    # number of elements for the calculation to work.
    cur_output = cur_output[cur_trg != pad_label]
    cur_trg = cur_trg[cur_trg != pad_label]



    # Now, we only care about the target label. For each value in both tensors, set the value to 
    # 1 if its one of the target labels, 0 otherwise.
    cur_output[cur_output != target_label] = 0
    cur_output[cur_output == target_label] = 1

    cur_trg[cur_trg != target_label] = 0
    cur_trg[cur_trg == target_label] = 1

    # Now we will go ahead and compute precision, recall and f1 score
    # First, let's get the number of True Positives, False Positives, and False Negatives
    tp = (cur_output * cur_trg).sum().int()
    fp = ((1 - cur_trg) * cur_output).sum().int()
    fn = (cur_trg * (1 - cur_output)).sum().int()

    return tp, fp, fn

def train(model, train_iterator, valid_iterator, criterion, optimizer, config, TTX, TRG, ASR, TTX_POS, ASR_POS, error_tags):
    wandb.watch(model, criterion, log="all", log_freq=10)
    N_EPOCHS = config.epochs
    CLIP = config.clip

    # Using the sweep dictionary, set the correct default for our sweep metric.
    if 'val_metric' not in wandb.config:
        # Send me an alert saying I forgot to set the metric (might get annoying, discuss at 12/6 meeting). Point is to allow training to continue in the case
        # of my first sweep that doesn't specify a metric, but the second sweep (and all following sweeps) will specify a metric.
        wandb.alert(title=f"Invalid Validation Metric for run {wandb.run.id}" , text="The validation metric you specified is not valid. Defaulting to loss.")
        best_valid_metric = float('inf')
        wandb.config['val_metric'] = 'loss'
        metric_did_improve = lambda x, y: x < y
    elif wandb.config['val_metric'] == 'loss':
        best_valid_metric = float('inf')
        metric_did_improve = lambda x, y: x < y
    elif wandb.config['val_metric'] == 'f1' or wandb.config['val_metric'] == 'precision' or wandb.config['val_metric'] == 'recall':
        best_valid_metric = float('-inf')
        metric_did_improve = lambda x, y: x > y
    else:
        wandb.alert(title=f"Invalid Validation Metric for run {wandb.run.id}" , text="The validation metric you specified is not valid. Defaulting to loss.")
        best_valid_metric = float('inf')
        wandb.config['val_metric'] = 'loss'
        metric_did_improve = lambda x, y: x < y

    best_epoch = 0
    example_ct = 0  # number of examples seen
    batch_ct = 0

    # Bug Fix: Basically there's a bug where if the model never acheives an F1 Score, it won't save the model. This results
    # in Wandb marking the model as a failure. This workaround ensures that when we get to the test phase, we have something
    # that runs.
    torch.save(model.state_dict(), MODEL_SAVE_FILENAME)

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        model.train()
        epoch_loss = 0
        batch_loss = 1e5

        for i, batch in enumerate(train_iterator):
            try:
                batch_loss, precision, recall, f1_score = train_batch(
                    model, batch, optimizer, criterion, CLIP, TTX, TRG, ASR, TTX_POS, ASR_POS, error_tags)
            except RuntimeError:
                print(
                    '\nRuntimeError! Skipping this batch, using previous loss as est\n')
                precision = np.nan
                recall = np.nan
                f1_score = np.nan
                
            example_ct += len(batch)
            batch_ct += 1

            # Report metrics every 25th batch
            if (batch_ct % 25) == 0:
                train_log(batch_loss, precision, recall, f1_score, example_ct, epoch)

            epoch_loss += batch_loss

        epoch_loss = epoch_loss / len(train_iterator)
        valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_iterator, criterion, TTX, TRG, ASR, error_tags)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Programatically set our validation metrics from wandb
        if wandb.config['val_metric'] == 'precision':
            val_metric = valid_precision
        elif wandb.config['val_metric'] == 'recall':
            val_metric = valid_recall
        elif wandb.config['val_metric'] == 'f1':
            val_metric = valid_f1
        else:
            val_metric = valid_loss


        if metric_did_improve(val_metric, best_valid_metric):
            best_valid_metric = val_metric
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_FILENAME)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {epoch_loss:.3f} | Train PPL: {np.exp(epoch_loss):7.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {np.exp(valid_loss):7.3f}')
        print(
            f'\t Val. Precision: {valid_precision:.3f} |  Val. Recall: {valid_recall:.3f} |  Val. F1: {valid_f1:.3f}')
        wandb.log({"epoch": epoch, 
                    "val_loss": valid_loss, 
                    "valid_precision": valid_precision, 
                    "valid_recall": valid_recall,
                    "valid_f1": valid_f1,
                    })

        if epoch - best_epoch >= config.early_stop:
            print(
                f'No improvement in {config.early_stop} epochs. Stopping training.\n')
            break

    return model, best_valid_metric


def train_batch(model, batch, optimizer, criterion, clip, TTX, TRG, ASR, TTX_POS, ASR_POS, error_tags: List[int]):
    ttx_src = batch.true_text
    asr_src = batch.asr
    trg = batch.tags
    ttx_pos = batch.true_text_pos
    asr_pos = batch.asr_pos

    optimizer.zero_grad()

    # TODO is cutting off part of TRG correct?
    output, _, _ = model(ttx_src, ttx_pos, asr_src, asr_pos, trg[:, :-1])

    print_debug_vals = np.random.randint(0, 40)
    # Print an example to the console, randomly
    if print_debug_vals == 1:
        print('TRUE TEXT: ', ' '.join(
            [TTX.vocab.itos[i] for i in ttx_src[0]]))
        print('ASR VERS.: ', ' '.join(
            [ASR.vocab.itos[i] for i in asr_src[0]]))
        print('TRUE TAGS: ', ' '.join([TRG.vocab.itos[i] for i in trg[0]]))
        print('MODEL OUT:  <sos>', ' '.join(
            [TRG.vocab.itos[np.argmax(i.tolist())] for i in output[0]]))
        print()

    # output = [batch size, ttx len - 1, output dim]
    # ttx_src = [batch size, ttx len]

    output_dim = output.shape[-1]

    output = output.contiguous().view(-1, output_dim)
    trg = trg[:, 1:].contiguous().view(-1)

    # Calculate the Recall, Precision and F1 Score for our error tags.
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for tag in error_tags:
        new_tp, new_fp, new_fn = get_positives_and_negatives(output, trg, TRG.vocab.stoi[tag], TRG.vocab.stoi['<pad>'])

        total_tp += int(new_tp)
        total_fp += int(new_fp)
        total_fn += int(new_fn)

    total_tp = torch.tensor(total_tp)
    total_fp = torch.tensor(total_fp)
    total_fn = torch.tensor(total_fn)

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1Score = 2 * (precision * recall) / (precision + recall)

    precision = precision.item()
    recall = recall.item()

    if torch.isnan(f1Score):
        f1Score = 0
    else:
        f1Score = f1Score.item()


    if print_debug_vals == 1:
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1 Score: ', f1Score)
    
    # Next, we'll go ahead and copy the tensor
    output_for_recall = output.clone()

    # Apply the argmax function to each inner tensor in the output tensor
    # Each index will now be the predicted tag value.
    output_for_recall = torch.argmax(output_for_recall, dim=1)

    # output = [batch size * trg len - 1, output dim]
    # trg = [batch size * trg len - 1]

    loss = criterion(output, trg)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    return loss.item(), precision, recall, f1Score


def train_log(loss, precision, recall, f1Score, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss, "train_precision": precision, "train_recall": recall, "train_f1": f1Score}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def evaluate(model, iterator, criterion, TTX, TRG, ASR, error_tags: List[int], print_outputs=False):
    model.eval()

    epoch_loss = 0
    precision = 0
    recall = 0
    f1_score = 0

    tps = 0
    fps = 0
    fns = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            ttx_src = batch.true_text
            asr_src = batch.asr
            trg = batch.tags
            ttx_pos = batch.true_text_pos
            asr_pos = batch.asr_pos

            # TODO is cutting off part of trg correct when not decoding trg?
            output, _, _ = model(ttx_src, ttx_pos, asr_src,
                                 asr_pos, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # ttx_src = [batch size, ttx len]

            output_dim = output.shape[-1]

            #             print('output shape:',output.shape)
            #             print('trg shape:',trg.shape)

            output_for_scoring = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            #             print('output shape:',output.shape)
            #             print('trg shape:',trg.shape)
            loss = criterion(output_for_scoring, trg)

            # Get the number of true positives, false positives, and false negatives
            # for this batch.
            new_tps = torch.tensor(0)
            new_fps = torch.tensor(0)
            new_fns = torch.tensor(0)

            for error_tag in error_tags:
                tag_tps, tag_fps, tag_fns = get_positives_and_negatives(output_for_scoring, trg, TRG.vocab.stoi[error_tag], TRG.vocab.stoi['<pad>'])

                new_tps += tag_tps
                new_fps += tag_fps
                new_fns += tag_fns

            if np.random.randint(0, 40) == 1 or print_outputs:
                print("VALIDATION OUTPUTS:")
                print('TRUE TEXT: ', ' '.join(
                    [TTX.vocab.itos[i] for i in ttx_src[0]]))
                print('ASR VERS.: ', ' '.join(
                    [ASR.vocab.itos[i] for i in asr_src[0]]))
                print('TRUE TAGS: ', ' '.join(
                    [TRG.vocab.itos[i] for i in batch.tags[0]]))
                print('MODEL OUT:  <sos>', ' '.join(
                    [TRG.vocab.itos[np.argmax(i.tolist())] for i in output[0]]))
                
                batch_precision = new_tps / (new_tps + new_fps)
                batch_recall = new_tps / (new_tps + new_fns)
                batch_f1Score = 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall)

                print("New TPS: ", new_tps)
                print("New FPs: ", new_fps)
                print("New FNs: ", new_fns)

                print(f"Precision of Batch: {batch_precision.item()}")
                print(f"Recall of Batch: {batch_recall.item()}")
                print(f"F1 Score of Batch: {batch_f1Score.item()}")
        
            new_tps = int(new_tps)
            new_fps = int(new_fps)
            new_fns = int(new_fns)

            epoch_loss += loss.item()
            tps += new_tps
            fps += new_fps
            fns += new_fns

    # Add together all the TPs, FPs and FNs from this batch to get our
    # reporting metrics. These don't need to be averaged out since they're applied
    # across the whole board.
    tps = torch.tensor(tps)
    fps = torch.tensor(fps)
    fns = torch.tensor(fns)

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return epoch_loss / len(iterator), precision.item(), recall.item(), f1_score.item()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def test(model, test_iterator, criterion, TTX, TRG, ASR, error_tags, model_filepath=MODEL_SAVE_FILENAME):
    model.load_state_dict(torch.load(model_filepath))

    test_loss, precision, recall, f1_score = evaluate(model, test_iterator, criterion,
                         TTX, TRG, ASR, error_tags, print_outputs=True)
    wandb.log({"test_loss": test_loss, "test_ppl": math.exp(test_loss), "test_precision": precision, "test_recall": recall, "test_f1": f1_score})

    artifact = wandb.Artifact('best_model', type='model')
    artifact.add_file(model_filepath)
    wandb.log_artifact(artifact)

    print(
        f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    return model, test_loss


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=600):
        # TODO make max length an input
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_pos, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = torch.arange(0, src_len).unsqueeze(
        #     0).repeat(batch_size, 1).to(self.device)
        pos = torch.tensor(src_pos)

        # pos = [batch size, src len]

        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class DimRedFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim * 2, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, 2 * hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Decoder_ttx(nn.Module):
    def __init__(self,
                 ttx_vocab_dim,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=600):
        super().__init__()

        self.device = device

        self.ttx_embedding = nn.Embedding(ttx_vocab_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer_ttx(hid_dim,
                                                      n_heads,
                                                      pf_dim,
                                                      dropout,
                                                      device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, ttx_src, ttx_enc_src, ttx_pos, asr_enc_src, ttx_mask, ttx_src_mask, asr_src_mask):
        # trg = [batch size, trg len]
        # ttx_src = [batch size, ttx src len]
        # enc_src = [batch size, src len, hid dim]
        # ttx_mask = [batch size, 1, ttx len, ttx len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = ttx_src.shape[0]
        ttx_len = ttx_src.shape[1]

        pos = torch.tensor(ttx_pos)
        # pos = torch.arange(0, ttx_len).unsqueeze(
        #     0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, ttx len]

        ttx_src = self.dropout(
            (self.ttx_embedding(ttx_src) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            ttx_src, ttx_attention, asr_attention = layer(ttx_src,
                                                          ttx_enc_src,
                                                          asr_enc_src,
                                                          ttx_mask,
                                                          ttx_src_mask,
                                                          asr_src_mask)

        # ttx_src = [batch size, ttx len, hid dim]
        # attention = [batch size, n heads, ttx len, src len]

        output = self.fc_out(ttx_src)

        # output = [batch size, ttx len, output dim]

        return output, ttx_attention, asr_attention


class Decoder_trg(nn.Module):
    def __init__(self,
                 ttx_vocab_dim,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=350):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer_trg(hid_dim,
                                                      n_heads,
                                                      pf_dim,
                                                      dropout,
                                                      device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, ttx_enc_src, asr_enc_src, trg_mask, ttx_src_mask, asr_src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, ttx_attention, asr_attention = layer(trg, ttx_enc_src, asr_enc_src, trg_mask, ttx_src_mask,
                                                      asr_src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, ttx_attention, asr_attention


class DecoderLayer_trg(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.dimred_feedforward = DimRedFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, ttx_enc_src, asr_enc_src, trg_mask, ttx_src_mask, asr_src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # ttx_encoder attention
        trg_ttx, ttx_attention = self.encoder_attention(
            trg, ttx_enc_src, ttx_enc_src, ttx_src_mask)

        # dropout, residual connection and layer norm
        trg_ttx = self.enc_attn_layer_norm(trg + self.dropout(trg_ttx))

        # trg_ttx = [batch size, trg len, hid dim]

        # asr_encoder attention
        trg_asr, asr_attention = self.encoder_attention(
            trg, asr_enc_src, asr_enc_src, asr_src_mask)

        # dropout, residual connection and layer norm
        trg_asr = self.enc_attn_layer_norm(trg + self.dropout(trg_asr))

        # trg_asr = [batch size, trg len, hid dim]

        trg = torch.cat((trg_ttx, trg_asr), 2)

        # trg = [batch size, trg len, 2 * hid dim]

        trg = self.dimred_feedforward(trg)

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, ttx_attention, asr_attention


class DecoderLayer_ttx(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.dimred_feedforward = DimRedFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ttx_src, ttx_enc_src, asr_enc_src, ttx_mask, ttx_src_mask, asr_src_mask):
        # ttx_src = [batch size, ttx len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # ttx_mask = [batch size, 1, ttx len, ttx len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _ttx_src, _ = self.self_attention(ttx_src, ttx_src, ttx_src, ttx_mask)

        # dropout, residual connection and layer norm
        ttx_src = self.self_attn_layer_norm(ttx_src + self.dropout(_ttx_src))

        # ttx_src = [batch size, ttx len, hid dim]

        # ttx_encoder attention
        ttx_ttx, ttx_attention = self.encoder_attention(
            ttx_src, ttx_enc_src, ttx_enc_src, ttx_src_mask)

        # dropout, residual connection and layer norm
        ttx_ttx = self.enc_attn_layer_norm(ttx_src + self.dropout(ttx_ttx))

        # ttx_ttx = [batch size, ttx len, hid dim]

        # asr_encoder attention
        ttx_asr, asr_attention = self.encoder_attention(
            ttx_src, asr_enc_src, asr_enc_src, asr_src_mask)

        # dropout, residual connection and layer norm
        ttx_asr = self.enc_attn_layer_norm(ttx_src + self.dropout(ttx_asr))

        # ttx_asr = [batch size, ttx len, hid dim]

        ttx_src = torch.cat((ttx_ttx, ttx_asr), 2)

        # ttx_src = [batch size, ttx len, 2 * hid dim]

        ttx_src = self.dimred_feedforward(ttx_src)

        # ttx_src = [batch size, ttx len, hid dim]

        # positionwise feedforward
        _ttx_src = self.positionwise_feedforward(ttx_src)

        # dropout, residual and layer norm
        ttx_src = self.ff_layer_norm(ttx_src + self.dropout(_ttx_src))

        # ttx_src = [batch size, ttx len, hid dim]
        # attention = [batch size, n heads, ttx len, src len]

        return ttx_src, ttx_attention, asr_attention


class Seq2Seq_ttx(nn.Module):
    def __init__(self,
                 ttx_encoder,
                 asr_encoder,
                 decoder,
                 ttx_src_pad_idx,
                 asr_src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.ttx_encoder = ttx_encoder
        self.asr_encoder = asr_encoder
        self.decoder = decoder
        self.ttx_src_pad_idx = ttx_src_pad_idx
        self.asr_src_pad_idx = asr_src_pad_idx
        #         self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_ttx_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.ttx_src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_asr_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.asr_src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_ttx_mask(self, ttx_src):
        # The mask for BPE ttx here should increment exposure of individual words.

        # ttx_src = [batch size, ttx len]

        ttx_pad_mask = (ttx_src != self.ttx_src_pad_idx).unsqueeze(
            1).unsqueeze(2)

        # ttx_src_pad_mask = [batch size, 1, 1, ttx len]

        ttx_len = ttx_src.shape[1]

        # ttx_sub_mask = torch.tril(torch.ones((ttx_len, ttx_len), device = self.device)).bool()
        ttx_sub_mask = torch.tril(torch.ones(
            (ttx_len + 1, ttx_len), device=self.device)).bool()
        ttx_sub_mask = ttx_sub_mask[1:, :]

        # ttx_sub_mask = [ttx len, ttx len]

        ttx_mask = ttx_pad_mask & ttx_sub_mask

        # ttx_mask = [batch size, 1, ttx len, ttx len]

        return ttx_mask

    def forward(self, ttx_src, asr_src, trg):
        # src = [batch size, src len]

        ttx_src_mask = self.make_ttx_src_mask(ttx_src)
        asr_src_mask = self.make_asr_src_mask(asr_src)
        ttx_mask = self.make_ttx_mask(ttx_src[:, :-1])

        # src_mask = [batch size, 1, 1, src len]
        # ttx_mask = [batch size, 1, ttx len, ttx len]

        ttx_enc_src = self.ttx_encoder(ttx_src, ttx_src_mask)
        asr_enc_src = self.asr_encoder(asr_src, asr_src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, ttx_attention, asr_attention = self.decoder(
            ttx_src[:, :-1],
            ttx_enc_src,
            asr_enc_src,
            ttx_mask,
            ttx_src_mask,
            asr_src_mask)

        # output = [batch size, ttx len, output dim]
        # attention = [batch size, n heads, ttx len, src len]

        return output, ttx_attention, asr_attention


class Seq2Seq_trg(nn.Module):
    def __init__(self,
                 ttx_encoder,
                 asr_encoder,
                 decoder,
                 ttx_src_pad_idx,
                 asr_src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.ttx_encoder = ttx_encoder
        self.asr_encoder = asr_encoder
        self.decoder = decoder
        self.ttx_src_pad_idx = ttx_src_pad_idx
        self.asr_src_pad_idx = asr_src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_ttx_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.ttx_src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_asr_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.asr_src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, ttx_src, ttx_pos, asr_src, asr_pos, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        ttx_src_mask = self.make_ttx_src_mask(ttx_src)
        asr_src_mask = self.make_asr_src_mask(asr_src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        ttx_enc_src = self.ttx_encoder(ttx_src, ttx_pos, ttx_src_mask)
        asr_enc_src = self.asr_encoder(asr_src, asr_pos, asr_src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, ttx_attention, asr_attention = self.decoder(trg,
                                                            ttx_enc_src,
                                                            asr_enc_src,
                                                            trg_mask,
                                                            ttx_src_mask,
                                                            asr_src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, ttx_attention, asr_attention


def translate_sentence(ttx_sentence, asr_sentence, ttx_src_field, asr_src_field, trg_field, model, device, max_len=50):
    model.eval()

    # Prepare true text

    if isinstance(ttx_sentence, str):
        ttx_tokens = tokenize_TTX(ttx_sentence)
    else:
        ttx_tokens = [token.lower() for token in ttx_sentence]

    ttx_tokens = [ttx_src_field.init_token] + \
        ttx_tokens + [ttx_src_field.eos_token]

    ttx_src_indexes = [ttx_src_field.vocab.stoi[token] for token in ttx_tokens]

    ttx_src_tensor = torch.LongTensor(ttx_src_indexes).unsqueeze(0).to(device)

    ttx_src_mask = model.make_ttx_src_mask(ttx_src_tensor)

    # Prepare ASR transcript

    if isinstance(asr_sentence, str):
        asr_tokens = tokenize_ASR(asr_sentence)
    else:
        asr_tokens = [token for token in asr_sentence]

    asr_tokens = [asr_src_field.init_token] + \
        asr_tokens + [asr_src_field.eos_token]

    asr_src_indexes = [asr_src_field.vocab.stoi[token] for token in asr_tokens]

    asr_src_tensor = torch.LongTensor(asr_src_indexes).unsqueeze(0).to(device)

    asr_src_mask = model.make_asr_src_mask(asr_src_tensor)

    with torch.no_grad():
        ttx_enc_src = model.ttx_encoder(ttx_src_tensor, ttx_src_mask)
        asr_enc_src = model.asr_encoder(asr_src_tensor, asr_src_mask)

    # trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    #     def forward(self, ttx_src, ttx_enc_src, asr_enc_src, ttx_mask, ttx_src_mask, asr_src_mask):

    ttx_mask = model.make_ttx_mask(ttx_src_tensor)

    with torch.no_grad():
        output, ttx_attention, asr_attention = model.decoder(ttx_src_tensor,
                                                             ttx_enc_src,
                                                             asr_enc_src,
                                                             ttx_mask,
                                                             ttx_src_mask,
                                                             asr_src_mask)

    trg_indexes = output.argmax(2)[0].tolist()

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens, ttx_attention, asr_attention


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
