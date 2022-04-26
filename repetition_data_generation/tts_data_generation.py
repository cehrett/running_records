# TTS Data Generation Pipeline
# Author: Porter Zach
import nemo
import nemo.collections.tts as nemo_tts
from text_mutation_generation import mutate_text
import torchaudio

# Spectrogram generators which take text as input and produce spectrogram
# See the Google Colab for this to find all available spectrograms
spec_gens = [
    (nemo_tts.models.Tacotron2Model.from_pretrained(
        model_name="Tacotron2-22050Hz").cuda(), "Tacotron2"),
]
print("Retrieving TTS models...")

# Spectrogram generators take text as input and produce spectrograms
# See the Google Colab for this to find all available spectrogram generators
tacotron = nemo_tts.models.Tacotron2Model.from_pretrained("tts_en_tacotron2")

# Vocoder models take spectrograms and produce actual audio
# See the Google Colab for this file to find all available vocoders
hifigan = nemo_tts.models.HifiGanModel.from_pretrained(
    model_name="tts_hifigan")

print("Done.")

# Converts text to speech using the text-spectrogram-audio process. Defaults to Tacotron2 and HifiGAN.
# Returns generated audio.


def text_to_audio_specv(text, spectrogram_generator=tacotron, vocoder=hifigan):
    parsed = spectrogram_generator.parse(text)
    spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    return audio.to('cpu').detach()

# Saves generated audio.


def save_audio(audio, file_name):
    rate = 22050  # Hz
    torchaudio.save(file_name, audio, rate)

# Utility function for processing text and downloading generated audio. Defaults to Tacotron2 and HifiGAN.


def process_and_save(text, file_name, spec_gen=tacotron, voc=hifigan):
    audio = text_to_audio_specv(text, spec_gen, voc)
    save_audio(audio, file_name)


if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog. Sally sells seashells by the sea shore."

    new_text = mutate_text(text)[0]

    print(new_text)
    process_and_save(new_text, "output.wav")
