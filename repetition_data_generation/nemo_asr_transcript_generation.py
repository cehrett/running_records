# This script is for generating asr transcripts of text strings using NeMo.



def get_asr_transcript(audio_filepaths:list):
    """
    Convert wav files to ASR transcripts using QuartzNet via NeMo.

    :param audio_filepaths: A list of strings, which gives the filepaths of the .wav files to be transcribed.
    :return: A list of strings, which is the ASR transcripts of each .wav file.
    """

    import nemo.collections.asr as nemo_asr

    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    transcripts = quartznet.transcribe(paths2audio_files=audio_filepaths)
    return transcripts


if __name__ == '__main__':
    import nemo.collections.asr as nemo_asr

    sample_files = ['output.wav']
    outputs = get_asr_transcript(sample_files)
    print(outputs)