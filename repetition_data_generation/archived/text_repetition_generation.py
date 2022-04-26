# This script instantiates the code written by Tori Vandermeulen and Carl Ehrett for the purpose of taking a line
# of text and generating repetitions within it (E.g. "Old man Moss got out of bed" -> "Old man Moss got out got out got
# "out of bed bed").

import numpy as np
import pandas as pd


def introduce_repetition(sentence: str, p=0.2):
    """
    This function takes as input a string and randomly introduces repetitions in it, tagging the result.

    :param sentence: A string containing one or more words
    :param p: a float between 0 and 1 (inclusive) which controls the probability of repetitions being introduced
    :return: a triple (n,wo_tags,rep_tags), where n is the new sentence, wo_tags is a vector with length equal to the
             number of words in n which describes the index in sentence of each word in n, and rep_tags is a vector of
             length equal to the number of words in sentence which describes for each word in sentence whether it is
             repeated in n.
    """
    sentence = sentence.split(' ')

    # low and high are the the limits of how many times the chunk of words will be repeated
    low = 2
    high = 4

    new_sent = []
    wo_tags = []
    rep_tag = []
    i = 0


    while i < len(sentence):

        coin_flip = np.random.binomial(1, p)  # If 1, we introduce a repetition.

        if coin_flip:

            rep = 0  # This counter will be used to add the desired number of repetitions to the new sentence
            rep_length = np.random.randint(low, high)
            if (len(sentence) - i) == 1:
                chunk_length = 1
            else:
                chunk_length = np.random.randint(1, len(sentence) - i)

            # Add repetition tages to rep_tag.
            begin_rep_tag = 'RB'+str(rep_length)
            inside_rep_tag = 'RI'+str(rep_length)
            rep_tag.append(begin_rep_tag)
            for k in range(0, chunk_length - 1):
                rep_tag.append(inside_rep_tag)

            while rep < rep_length:  # This loop will add the repetition to the new sentence rep_length times.

                for j in range(0, chunk_length):
                    new_sent.append(sentence[i + j])
                    wo_tags.append(i + j)

                rep += 1
            i += chunk_length

        else:
            new_sent.append(sentence[i])
            wo_tags.append(i)
            rep_tag.append('O')
            i += 1

    return new_sent, wo_tags, rep_tag


def create_text_repetition_data(sentence: str):
    """
    This function creates a pandas dataframe fit for NER. Each row of the df
    corresponds to one word from the original sentence. The columns are: word,
    and tag. Tag is an ordered pair, of the repetition chunk length and the
    number of times the chunk is repeated.
    """
    repeated_sentence, _, repetition_tags = introduce_repetition(sentence)

    sentence = sentence.split(' ')

    try:
        df = pd.DataFrame(data={'word': sentence, 'tag': repetition_tags})
    except:
        print(sentence)
        print(repetition_tags)
        print(repeated_sentence)
        Exception

    return repeated_sentence, df


def create_text_repetition_dataset(list_of_sentences):
    """
    This function takes as input a list of sentences, and outputs a df fit for NER
    """

    dfs = []  # This list will store all dfs for individual sentences

    for idx, sentence in enumerate(list_of_sentences):
        _, sentence_df = create_text_repetition_data(sentence)
        sentence_df['sentence'] = idx
        dfs.append(sentence_df)

    return pd.concat(dfs)


if __name__ == '__main__':
    print(create_text_repetition_dataset(['Test sentence one', 'Test sentence two', 'Test sentence three']))
