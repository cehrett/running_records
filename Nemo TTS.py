from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
import sys
import numpy as np
import random

def convert_to_audio(sentence):
  supported_spec_gen = ["tacotron2", "glow_tts","random"]
  supported_audio_gen = ["waveglow", "squeezewave", "uniglow", "melgan", "hifigan", "two_stages","random"]

  print("Choose one of the following spectrogram generators:")
  print([model for model in supported_spec_gen])
  spectrogram_generator = input()
  if spectrogram_generator not in supported_spec_gen or spectrogram_generator == 'random':
    if spectrogram_generator not in supported_spec_gen:
      print('Invalid Input, selecting random model.')
    spectrogram_generator = random.choice([x for x in supported_spec_gen if x != "random"])

  print("Choose one of the following audio generators:")
  print([model for model in supported_audio_gen])
  audio_generator = input()

  if audio_generator not in supported_spec_gen or audio_generator == 'random':
    if audio_generator not in supported_audio_gen:
      print('Invalid Input, selecting random model.')
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
  print('spec gen' + spectrogram_generator)
  print('audio gen' + audio_generator)

  spec_generator = SpectrogramGenerator.from_pretrained(pretrained_model_spec, override_config_path=override_conf)

  # choosing specified vocoder model
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
      if mel2spec == "encoder_decoder":
          from nemo.collections.tts.models.ed_mel2spec import EDMel2SpecModel

          pretrained_mel2spec_model = "EncoderDecoderMelToSpec-22050Hz"
          mel2spec_model = EDMel2SpecModel.from_pretrained(pretrained_mel2spec_model)
          vocoder.set_mel_to_spec_model(mel2spec_model)

      if linvocoder == "degli":
          from nemo.collections.tts.models.degli import DegliModel

          pretrained_linvocoder_model = "DeepGriffinLim-22050Hz"
          linvocoder_model = DegliModel.from_pretrained(pretrained_linvocoder_model)
          vocoder.set_linear_vocoder(linvocoder_model)

      TwoStagesModel = True
  else:
      raise NotImplementedError
  # Download and load the pretrained waveglow model
  #if not TwoStagesModel:
  vocoder = Vocoder.from_pretrained(pretrained_model_vocoder)
  # All spectrogram generators start by parsing raw strings to a tokenized version of the string
  parsed = spec_generator.parse(sentence)

  # They then take the tokenized string and produce a spectrogram
  spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
  # Finally, a vocoder converts the spectrogram to audio
  audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
  if audio_generator == "hifigan" or audio_generator == "hifigan" or audio_generator =="two_stages":
    audio = audio.detach().numpy()
  print(audio)

  return audio

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    convert_to_audio(*sys.argv[1:])
