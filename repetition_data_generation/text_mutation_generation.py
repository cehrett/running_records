# Sentence Mutations: Repetitions, Substitutions, & Deletions
# Author: Porter Zach

import random
import numpy as np
import string

# The formal tags
# This tagging method causes problems when repetitions and deletions interact. Index tags should likely be used instead.
deletion_tag = "-"
repetition_beginning_tag = "RB"
repetition_continue_tag = "RI"
correct_tag = "O"

# Authors: Tori Vandermeulen and Carl Ehrett


def add_repetitions(sentence: str, p=0.2, max_repeats=3, repeat_token=",,,"):
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

    new_sent = []
    wo_tags = []
    rep_tag = []
    i = 0

    while i < len(sentence):

        # If 1, we introduce a repetition.
        coin_flip = np.random.binomial(1, p)

        if coin_flip:

            rep = 0  # This counter will be used to add the desired number of repetitions to the new sentence
            rep_length = np.random.randint(2, max_repeats)
            if (len(sentence) - i) == 1:
                chunk_length = 1
            else:
                chunk_length = np.random.randint(1, len(sentence) - i)

            # Add repetition tages to rep_tag.
            begin_rep_tag = repetition_beginning_tag + str(rep_length)
            inside_rep_tag = repetition_continue_tag + str(rep_length)
            rep_tag.append(begin_rep_tag)
            for k in range(0, chunk_length - 1):
                rep_tag.append(inside_rep_tag)

            # This loop will add the repetition to the new sentence rep_length times.
            while rep < rep_length:

                for j in range(0, chunk_length):
                    token = repeat_token
                    if j < chunk_length - 1 or rep == rep_length - 1:
                        token = ""
                    new_sent.append(sentence[i + j] + token)

                    wo_tags.append(i + j)

                rep += 1
            i += chunk_length

        else:
            new_sent.append(sentence[i])
            wo_tags.append(i)
            rep_tag.append(correct_tag)
            i += 1

    return new_sent, wo_tags, rep_tag

# gets the substitution for a word given the original.


def get_substitution(original_word):
    # TODO: Finish this function
    # randomly decide between a semantic or syntactic substitution
    # for semantic (definition similar) subs, we can use w2vec
    # for syntactic (sounds similar) subs... find words with low DL distances somehow?
    return "SUB"


def add_substitutions(text, prob=0.2):
    # split the sentence into a list of words
    words = text.split(' ')
    # initialize the resulting sentence and tags lists
    new_sent = []
    index_tags = []
    sub_tags = []
    # for each word
    for i in range(len(words)):
        # if this word is chosen to be substituted
        if random.random() < prob:
            # add the substituted word to the new sentence
            new_sent.append(get_substitution(words[i]))
            # add the substitution to the formal tags list
            sub_tags.append(new_sent[-1])
            # add the substitution index to the index tags (-1 is used to represent a substitution)
            index_tags.append(-1)
        # if the word isn't substituted, add the default items to the lists
        else:
            new_sent.append(words[i])
            index_tags.append(i)
            sub_tags.append(correct_tag)
    return new_sent, index_tags, sub_tags


def add_deletions(text, prob=0.2):
    # split the sentence into a list of words
    words = text.split(' ')
    # initialize the resulting sentence and tags lists
    new_sent = []
    index_tags = []
    del_tags = []
    # for each word
    for i in range(len(words)):
        # if this word is chosen to be deleted
        if random.random() < prob:
            # don't add it to the resulting words or index list
            # add the omission/deletion tag to the formal tags list
            del_tags.append(deletion_tag)
            # if the word isn't deleted, add the default items to the lists
        else:
            new_sent.append(words[i])
            index_tags.append(i)
            del_tags.append(correct_tag)
    return new_sent, index_tags, del_tags

# Performs deletions, repetitions, and substitutions on a given text
# Returns the modified text and the index tags for that text


def mutate_text(text, del_prob=0.2, rep_prob=0.2, max_reps=3, sub_prob=0.2, remove_punc=False):
    # TODO: Handle the Special Tags Generated
    # Introduce deletions
    new_text, index_tags, _ = mutate_selectively(
        text, "del", del_prob, rep_prob, max_reps, sub_prob, remove_punc)

    # Introduce repetitions
    new_text, index_tags_new, _ = mutate_selectively(
        new_text, "rep", del_prob, rep_prob, max_reps, sub_prob, remove_punc)
    index_tags = get_new_index_tags(index_tags, index_tags_new)

    # Introduce substitutions
    new_text, index_tags_new, _ = mutate_selectively(
        new_text, "sub", del_prob, rep_prob, max_reps, sub_prob, remove_punc)
    index_tags = get_new_index_tags(index_tags, index_tags_new)

    return new_text, index_tags

# Allows caller to choose which mutation to make


def mutate_selectively(text, mutation, del_prob=0.2, rep_prob=0.2, max_reps=3, sub_prob=0.2, remove_punc=False):
    new_text = ""
    tags = []

    # For any mutation, get the text and index tags
    # Create a single string from the string list
    # and save the tags

    if mutation == "rep":
        repetitions = add_repetitions(text, rep_prob, max_reps)
        new_text = " ".join(repetitions[0])
        tags = repetitions[1]
        special_tags = repetitions[2]
    elif mutation == "del":
        deletions = add_deletions(text, del_prob)
        new_text = " ".join(deletions[0])
        tags = deletions[1]
        special_tags = deletions[2]
    elif mutation == "sub":
        substitutions = add_substitutions(text, sub_prob)
        new_text = " ".join(substitutions[0])
        tags = substitutions[1]
        special_tags = substitutions[2]

    if remove_punc:
        new_text = remove_punctuation(new_text)

    return new_text, tags, special_tags


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Gets the correct index tags for a newly modified string given the old
# index tags and the new tags based on the new modifications.


def get_new_index_tags(original_tags, new_wrong_tags):
    correct_tags = []
    for i in range(len(new_wrong_tags)):
        new_index = new_wrong_tags[i]
        if new_index == -1:
            correct_tags.append(-1)
        else:
            correct_tags.append(original_tags[new_index])
    return correct_tags


if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog. Sally sells seashells by the sea shore."

    print(f"Original sentence: {text}")
    print("\nSentence with mutations:")
    mutated, index_tags = mutate_text(text)
    print(mutated)
    print(index_tags)
    print("\nNew sentence without punctuation:")
    no_punc = remove_punctuation(mutated)
    print(no_punc)
