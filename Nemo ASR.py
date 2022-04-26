# %%

# NeMo's "core" package
import sys

print(sys.version)
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr
import json
import librosa
import os
import pandas as pd
from pydub import AudioSegment


# %% md

# Making JSON Manifest Files for Nemo Training

# %%

# --- Building Manifest Files --- #

# Function to build a manifest
def build_manifest_df(df, manifest_path, root_of_data='.\\data\\', Windows = True):
    with open(manifest_path, 'w') as fout:
        for index, row in df.iterrows():
            if Windows:
                row.filepath = '\\'.join(row.filepath.split('/'))
                delimeter = '\\'
            else:
                delimeter = '/'
            # mp3 to wav
            mp3sound = AudioSegment.from_mp3(root_of_data + str(row.filepath).rsplit(delimeter, 1)[1])
            wav_path = (root_of_data + (str(row.filepath)).strip('.mp3') + '.wav').rsplit(delimeter, 1)[1]
            mp3sound.export(root_of_data + wav_path, format="wav")
            duration = librosa.core.get_duration(filename=root_of_data + wav_path)
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": root_of_data + wav_path,
                "duration": duration,
                "text": row['Cleaned audio transcript']
            }
            json.dump(metadata, fout)
            fout.write('\n')


# %%

# Using audio data spreadsheet to build the manifests
print(os.getcwd())
df = pd.read_csv('.\\data\\running_records_audio_info.csv')

# %%

train_df = df[0:22].copy()
valid_df = df[23:36].copy()

# %%

build_manifest_df(train_df, './data/train_manifest_all.json')
build_manifest_df(valid_df, './data/test_manifest_all.json')
# %% md

# Out of Box Model Usage

# %%

# scr
# This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# %%

# scr
quartznet.transcribe(paths2audio_files=['.\\data\\ClarkSusan.wav'])

# %%
# Get intermediate output from Quartznet model

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model = quartznet
model.decoder.decoder_layers.register_forward_hook(get_activation('decoder_layers'))
x=['.\\data\\CollinsCole.wav']
output = model.transcribe(paths2audio_files=x)
print(activation['decoder_layers'])

# %%

# Listen to the audio file we're transcribing:
from pydub.playback import play

audiofile = '..\\data\\ClarkSusan.wav'
AudioSegment.from_mp3(audiofile)

# %% md

# Training (in progress)

# %%

# Need to git clone NeMo only once. Don't include it in (our) git repository.
# !git clone https://github.com/NVIDIA/NeMo/

# %%

# --- Config Information ---#
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML
# config_path = 'quartznet_15x5.yaml'

yaml = YAML(typ='safe')
with open('../NeMo/examples/asr/conf/quartznet_15x5.yaml') as f:
    params = yaml.load(f)
print(params)

# %%

import pytorch_lightning as pl

# %%

trainer = pl.Trainer(gpus=0, max_epochs=5)

# %%

train_mainfest = '../data/train_manifest_all.json'
test_manifest = '../data/test_manifest_all.json'

# %%

params['model']

# %%

params['model']['train_ds']['max_duration'] = 500

# %%

from omegaconf import DictConfig

params['model']['train_ds']['manifest_filepath'] = '../data/train_manifest_all.json'
params['model']['validation_ds']['manifest_filepath'] = '../data/test_manifest_all.json'
first_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

# %%

# Start training!!!
trainer.fit(first_asr_model)

# %%
