import argparse
from text_mutation_generation import *
import pandas as pd
import traceback
import re
from tqdm import tqdm
from typing import List, Tuple


def getSentencesToBeProcessed(input_file: str) -> List[str]:
    """
    This function will read from the input_file path passed in and return
    a list of all the sentences to be read. Each sentence will have whitespace
    stripped from the end.

    Parameters
    ----------
    input_file : str
        The path to the file containing a list of sentences we are interested
        in processing. The file should contain one sentence on each line.

    Returns
    -------
    List[str]
        A list of all the sentences in the input file..
    """
    with open(input_file, 'r') as f:
        allSentences = []
        for line in f:
            allSentences.append(line.strip())

    return allSentences


def getCleanedSentences(sentences: List[str], goal_length: int) -> Tuple[List[str], List[str]]:
    """
    This function will take a list of sentences and return two lists. The first
    list will contain goal_length sentences that hav ebeen cleaned for processing,
    while the second list will contain the sentences that will not be ready for
    processing.

    Parameters
    ----------
    sentences : List[str]
        A list of sentences to be cleaned.
    goal_length : int
        The length of the sentences we want to return.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists. The first list contains the goal_length
        sentences that have been cleaned for processing, while the second list
        contains the sentences that will not be ready for processing.
    """
    print(f"Getting {goal_length} clean sentences for processing.")
    cleanedSentences = []

    while len(cleanedSentences) < goal_length:
        currentClean = re.sub(
            "-*\s*\([A-Z][A-Z]\)\s*", "", sentences.pop(0).strip()).strip()
        currentClean = re.sub("\s*-\s", " ", currentClean).strip()
        if re.search("[0-9]+.", currentClean) or currentClean.isspace() or len(currentClean) == 0:
            continue
        cleanedSentences.append(currentClean)

    assert len(
        cleanedSentences) == goal_length, f'Expected number of cleaned sentences to be {goal_length}, got {len(cleanedSentences)}'

    return cleanedSentences, sentences


def processSentences(sentences: List[str], output_file_name: str, input_file: str, max_length: int = 1000) -> None:
    """
    This function will process the sentences in the list and write them to 3 output
    files. One output file will contain sentences with substitution mutations,
    one will contain sentences with deletion mutations, and one will contain
    sentences with repetition mutations. The naming convention for the output files
    will be(output_file_name + "_mutation.csv"). The output files will contain
    4 columns: original text, mutated texts and the index tags and mutation tags.

    Parameters
    ----------
    sentences: List[str]
        A list of sentences to be processed.
    output_file_name: str
        The name of the output file.
    input_file: str
        The path to the input file. We will rewrie this file with all of


    """
    cols = ['original_text', 'mutated_text', 'index_tags', 'mutation_tags']

    try:
        subs_df = pd.read_csv(output_file_name + "_substitutions.csv")
        dels_df = pd.read_csv(output_file_name + "_deletions.csv")
        reps_df = pd.read_csv(output_file_name + "_repetitions.csv")
    except FileNotFoundError:
        print("No output file found. Creating new output file.")
        subs_df = pd.DataFrame(
            columns=cols)
        dels_df = pd.DataFrame(
            columns=cols)
        reps_df = pd.DataFrame(
            columns=cols)

    # Get max_length sentences to process. We need them to be clean ones.
    cleanedSentences, notCleanedSentences = getCleanedSentences(
        sentences, max_length)

    with open(input_file, 'w') as f:
        for sentence in tqdm(notCleanedSentences, desc='Writing remaining sentences back to input file.'):
            f.write(sentence + '\n')

    for cleanedSentence in tqdm(cleanedSentences, desc='Mutating Sentences'):
        try:
            # Keep Generating Deletion Permutations Until we get one
            # with at least one valid word.
            del_tags = []
            while 'O' not in del_tags:
                new_sentence_del, del_index_tags, del_tags = mutate_selectively(
                    cleanedSentence, "del", del_prob=0.2, remove_punc=False)
            dels_df = pd.concat([dels_df, pd.DataFrame(
                {'original_text': [cleanedSentence], 'mutated_text': [new_sentence_del], 'index_tags': [del_index_tags], 'mutated_tags': [del_tags]})])

            # Get Mutated Sentence with Repetititon
            new_sentence_rep, rep_index_tags, rep_tags = mutate_selectively(
                cleanedSentence, "rep", rep_prob=0.2, max_reps=3, remove_punc=False)
            reps_df = pd.concat([reps_df, pd.DataFrame(
                {'original_text': [cleanedSentence], 'mutated_text': [new_sentence_rep], 'index_tags': [rep_index_tags], 'mutated_tags': [rep_tags]})])

            # Get Mutated Sentence with Substitutions
            new_sentence_subs, sub_index_tags, sub_tags = mutate_selectively(
                cleanedSentence, "sub", sub_prob=0.2, remove_punc=False)

            # Check that the number of words in new_sentence_subs is the same
            # as the number of tags in sub_tags
            if len(new_sentence_subs.split()) != len(sub_index_tags):
                raise Exception(
                    "Number of words in new_sentence_subs and sub_tags do not match.")

            subs_df = pd.concat([subs_df, pd.DataFrame(
                {'original_text': [cleanedSentence], 'mutated_text': [new_sentence_subs], 'index_tags': [sub_index_tags], 'mutated_tags': [sub_tags]})])
        except IndexError:
            traceback.print_exc()
            print(sentence)
            continue

    subs_df.to_csv(output_file_name + "_substitutions.csv",
                   index=False)
    dels_df.to_csv(output_file_name + "_deletions.csv",
                   index=False)
    reps_df.to_csv(output_file_name + "_repetitions.csv",
                   index=False)


if __name__ == '__main__':
    # Arg Parsing. This lets us go from the command line.
    # Demo: python create_training_data.py --input_file data/unprocessedSentences.txt --output_file data/output.txt --max_length=5
    parser = argparse.ArgumentParser(
        description='Create training data for the repetition task.')
    parser.add_argument('--input_file', type=str,
                        help='Path to the input file.', required=True)
    parser.add_argument('--output_file', type=str,
                        help='Path to the output file.', required=True)
    parser.add_argument('--max_length', type=int,
                        help='Maximum number of lines to parse.', default=None)

    args = parser.parse_args()
    sentences = getSentencesToBeProcessed(args.input_file)
    processed_sentences = processSentences(
        sentences, args.output_file, args.input_file, args.max_length)
