
import numpy as np
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
import random

def convert_to_audio(sentence, spectrogram_generator='random', audio_generator='random'):
  supported_spec_gen = ["tacotron2", "glow_tts","random"]
  supported_audio_gen = ["waveglow", "squeezewave", "uniglow", "melgan", "hifigan", "two_stages","random"]

  # randomly chooses model types if not specified in function call
  if spectrogram_generator == 'random':
    spectrogram_generator = random.choice([x for x in supported_spec_gen if x != "random"])

  if audio_generator == 'random':
    audio_generator = random.choice([x for x in supported_audio_gen if x != "random"])
    
  # choosing specified spectrogram model
  override_conf = None
  if spectrogram_generator == "tacotron2":
      from nemo.collections.tts.models import Tacotron2Model

      pretrained_model_spec = "tts_en_tacotron2"

  elif spectrogram_generator == "glow_tts":
      from nemo.collections.tts.models import GlowTTSModel

      pretrained_model_spec = "tts_en_glowtts"
      import wget
      from pathlib import Path

      if not Path("cmudict-0.7b").exists():
          filename = wget.download("http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b")
          filename = str(Path(filename).resolve())
      else:
          filename = str(Path("cmudict-0.7b").resolve())
      conf = SpectrogramGenerator.from_pretrained(pretrained_model_spec, return_config=True)

      if "params" in conf.parser:
          conf.parser.params.cmu_dict_path = filename
      else:
          conf.parser.cmu_dict_path = filename
      override_conf = conf
  else:
      raise NotImplementedError

  # final spectrogram generator
  spec_generator = SpectrogramGenerator.from_pretrained(pretrained_model_spec, override_config_path=override_conf)

  # choosing specified vocoder (audio) model
  RequestPseudoInverse = False
  TwoStagesModel = False

  if audio_generator == "waveglow":
      from nemo.collections.tts.models import WaveGlowModel

      pretrained_model_vocoder = "tts_waveglow_88m"
  elif audio_generator == "squeezewave":
      from nemo.collections.tts.models import SqueezeWaveModel

      pretrained_model_vocoder = "tts_squeezewave"
  elif audio_generator == "uniglow":
      from nemo.collections.tts.models import UniGlowModel

      pretrained_model_vocoder = "tts_uniglow"
  elif audio_generator == "melgan":
      from nemo.collections.tts.models import MelGanModel

      pretrained_model_vocoder = "tts_melgan"
  elif audio_generator == "hifigan":
      from nemo.collections.tts.models import HifiGanModel

      pretrained_model_vocoder = "tts_hifigan"

  elif audio_generator == "two_stages":
      from nemo.collections.tts.models import TwoStagesModel

      cfg = {'linvocoder': {'_target_': 'nemo.collections.tts.models.two_stages.GriffinLimModel',
                            'cfg': {'n_iters': 64, 'n_fft': 1024, 'l_hop': 256}},
            'mel2spec': {'_target_': 'nemo.collections.tts.models.two_stages.MelPsuedoInverseModel',
                          'cfg': {'sampling_rate': 22050, 'n_fft': 1024,
                                  'mel_fmin': 0, 'mel_fmax': 8000, 'mel_freq': 80}}}
      vocoder = TwoStagesModel(cfg)
     
      TwoStagesModel = True
  else:
      raise NotImplementedError
  if not TwoStagesModel:
    vocoder = Vocoder.from_pretrained(pretrained_model_vocoder)
  # All spectrogram generators start by parsing raw strings to a tokenized version of the string
  parsed = spec_generator.parse(sentence)

  # They then take the tokenized string and produce a spectrogram
  spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
  # Finally, a vocoder converts the spectrogram to audio
  audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
  
  # converting audio type to a numpy array if needed
  try:
    audio = audio.detach().numpy()
  except:
    print('Audio already is numpy array')
  print('Spectrogram Generator: ' + spectrogram_generator)
  print('Audio Generator: ' + audio_generator)

  return audio

if __name__ == '__main__':
    audio = convert_to_audio('Testing testing testing')
    print(audio)