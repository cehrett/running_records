# This script defines a function which takes as input a sentence and delivers as output a reading of that sentence both
# as text and as audio, along with tags describing the repetitions in terms of the indices of the words in the
# original sentence.

def create_repetition_audio_text_and_tags(sentence: str, spectrogram_generator = 'glow_tts'):
    """
    Takes as input a sentence and delivers as output a reading of that sentence both as text and as audio,
    along with tags describing the repetitions in terms of the indices of the words in the original sentence.
    :param spectrogram_generator: Tells which NeMo spectrogram generator to use.
    :param sentence:str
    :return: triple (new_text,tags,audio) where new_text is the new text, tags is a df describing the repetitions,
    and audio is the audio of the repetition-reading.
    """

    from running_records.repetition_data_generation.text_repetition_generation import create_text_repetition_data
    from running_records.repetition_data_generation import tts_data_generation

    params = {
        'spectrogram_generator': spectrogram_generator,  # "tacotron2" or "glow_tts"
        'audio_generator': "waveglow"
    }

    new_text, tags = create_text_repetition_data(sentence)
    audio = tts_data_generation.convert_to_audio(' '.join(new_text), params)
    return new_text, tags, audio


def create_dataset(sentences: list, filename: str, root='.\\data\\', start_idx=0, save_csv=False):
    """
    Creates a dataset of repetition-laden readings of sentences, complete with tags.

    :param sentences: A list of strings, the sentences to be read and tagged.
    :param filename: The filename under which the resulting csv is to be written. Do not include .csv extension. This will also be used for the .wav files.
    :param root: The location where the csv and audio files will be placed.
    :return:
    """
    true_text = []
    tags = []
    rep_audio_filenames = []
    import soundfile as sf
    from running_records.repetition_data_generation.nemo_asr_transcript_generation import get_asr_transcript
    import pandas as pd
    import gc
    import torch

    for i, sentence in enumerate(sentences):
        # Try to free up CUDA memory
        gc.collect()
        torch.cuda.empty_cache()

        rep_text, rep_tags, rep_audio = create_repetition_audio_text_and_tags(sentence)
        true_text.append(sentence)
        tags.append(rep_tags.tag.values)

        # Write audio to wav files
        rep_audio_filename = root + 'audio\\' + filename+str(i+start_idx)+'.wav'
        sf.write(rep_audio_filename, rep_audio.to('cpu').detach().numpy()[0], 22050)
        rep_audio_filenames.append(rep_audio_filename)

    # Get ASR transcript
    rep_asr = get_asr_transcript(rep_audio_filenames)

    # Make pandas DF of true texts, tags, audio filenames, and asr transcripts.
    data = {'true_text':true_text, 'tags': tags, 'filepath': rep_audio_filenames, 'asr': rep_asr}
    rep_df = pd.DataFrame(data=data)

    # Save DF to csv
    if save_csv:
        csv_filename = root + filename + '.csv'
        rep_df.to_csv(csv_filename)
    else:
        return rep_df


if __name__ == '__main__':
    import pandas as pd
    import soundfile as sf
    from running_records.repetition_data_generation.nemo_asr_transcript_generation import get_asr_transcript

    train_sentences = ['Of the training sentences included in this dataset, this one is the first.',
                       'There are multiple sentences in this dataset: for example, this one contains one contains one '
                       'contains a repetition.',
                       'Some days you get the bear, and some days, well, some days... oh boy.',
                       'The previous sentence was a notable non sequitur of a particularly ursine flavor.',
                       'This is explicitly not a coded warning that there is a bear standing behind you right now.']

    valid_sentences = ['The sentences listed here are not valid sentences in the sense that the sentences are valid, '
                       'but rather in the sense that the sentences are for the validation set.',
                       'Quite a long one, that first validation sentence was.',
                       'Without a pair of boots, stomping capacity is typically diminished.',
                       'Where some books are about spiders, other books are instead about woodworking, and a subset '
                       'of these books have spiders physically present on top of them.']

    test_sentences = ['Here is one test sentence',
                      'Oh I guess this is a second test sentence!',
                      "Well, what do you know - it's a third test sentence.",
                      "This fourth test sentence might be the best one yet, don't you think?",
                      "Never mind, this fifth and final test sentence is pretty obviously the greatest of all time."]

    create_dataset(train_sentences, 'train_sentence', save_csv=True)
    create_dataset(valid_sentences, 'valid_sentence', save_csv=True)
    create_dataset(test_sentences, 'test_sentence', save_csv=True)
