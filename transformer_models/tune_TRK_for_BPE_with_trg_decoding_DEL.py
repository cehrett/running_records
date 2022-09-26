from mltb.hyperopt import fmin
from hyperopt import hp, tpe, STATUS_OK
from hyperopt.pyll import scope

import transformer_repetition_kit as trk
import random
import numpy as np
import torch
from math import gcd
import time

import argparse
 
# Parse command line arguments
parser = argparse.ArgumentParser(description="Use hyperopt to tune the transformer model.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data", help="data file path")
parser.add_argument("-m", "--max_evals", help="number of hyperopt trials")
parser.add_argument("-b", "--use_bpe", help="Use byte-pair encoding", nargs='?', const=1, default=False)
args = parser.parse_args()
config = vars(args)
print('config:',config)

# Set number of evals that Hyperopt can perform
max_evals = int(config['max_evals'])

# Get whether to use bpe
bpe = config['use_bpe']

# Set filepaths
ASR_df_filepath = config['data']
asr_text_filepath = 'asr.txt'
ttx_text_filepath = 'ttx.txt'
train_filename = 'train_sentence.csv'
valid_filename = 'valid_sentence.csv'
test_filename = 'test_sentence.csv'
trials_filename = f'{config["data"].split("/")[-1].split(".")[0]}.trials'

# Load data
trk.load_data(ASR_df_filepath = ASR_df_filepath,
			train_filename = train_filename,
			valid_filename = valid_filename,
			test_filename = test_filename,
			asr_text_filepath = asr_text_filepath,
			ttx_text_filepath = ttx_text_filepath)

# Define search space
space = {
	'epochs': 256,
	'batch_size': hp.choice('batch_size',[64,128]),
	'learning_rate': hp.lognormal('learning_rate',-6,1),
	'dataset': ASR_df_filepath,
	'hid_dim_nheads_multiplier': scope.int(hp.quniform('hid_dim_nheads_multiplier',6,50,4)),
	'enc_layers': scope.int(hp.quniform('enc_layers',2,8,1)),
	'dec_layers': scope.int(hp.quniform('dec_layers',2,8,1)),
	'enc_heads': scope.int(hp.quniform('enc_heads',2,12,2)),
	'dec_heads': scope.int(hp.quniform('dec_heads',2,12,2)),
	'enc_pf_dim': scope.int(hp.qloguniform('enc_pf_dim',3,7,40)),
	'dec_pf_dim': scope.int(hp.qloguniform('dec_pf_dim',3,7,40)),
	'enc_dropout': hp.lognormal('enc_dropout',-2.5,1),
	'dec_dropout': hp.lognormal('dec_dropout',-2.5,1),
	'clip': 1,
	'bpe_vocab_size': scope.int(hp.qloguniform('bpe_vocab_size',6,8,200)),
	'decode_trg': True,
	'early_stop': 128,
	'overfit': False # Used for dev, set to false for real training
}

# Define objective function
def objective(params):
	print(f'PARAMS: {params}')
	
	# Fix problem of dropouts being above one
	params['enc_dropout'] = min(1,params['enc_dropout'])
	params['dec_dropout'] = min(1,params['dec_dropout'])

	loss, model = hyperopt_train_test(params)
	print(f'     LOSS: {loss}')
	return {
	'loss': loss,
	'status': STATUS_OK,
	# -- store other results like this
	'eval_time': time.time(),
	'params': params#,
	# 'model': model # Commented this out due to pickle woes
	}

# Need to get lcm of enc_heads later
def lcm(a, b):
	return abs(a*b) // gcd(a, b)

# Define hyperopt training function
def hyperopt_train_test(params):
	# Create tokenizer
	if bpe: 
		tokenizer = trk.create_train_bpe_tokenizer(params['bpe_vocab_size'],
			asr_text_filepath = \
			asr_text_filepath,
			ttx_text_filepath = ttx_text_filepath,
			save_tokenizer = False,
			tokenizer_filename = "./tokenizer-test.json"
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
	if bpe:
		output = tokenizer.encode("Hello, y'all! How are you 😁 ? [WSP]")
		print(output.tokens)
		print(output.ids)
	
	device = torch.device('cuda')
	
	# Update params
	params['hid_dim'] = lcm(params['enc_heads'],params['dec_heads']) * params['hid_dim_nheads_multiplier']
	
	# Train the model and get the loss
	model, train_loss, test_loss = trk.model_pipeline(params, 
								device,
								train_data,
								valid_data,
								test_data,
								TTX,
								TRG,
								ASR
								)
	
	if params['overfit']: loss = train_loss
	else: loss = test_loss
	
	return loss, model

# Minimize objective function
print('Beginning model search and tuning.')
best, trials = fmin(objective,
	space=space,
	algo=tpe.suggest,
	max_evals=max_evals,
	filename=trials_filename)
print(f'Best model: {best}')
print(f'Number of trials: {len(trials.trials)}')
print(f'Trials saved as {trials_filename}')

# best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model'] # Commented this out due to pickle woes

# torch.save(best_model.state_dict(), 'best_model.pt') # Commented this out due to pickle woes
