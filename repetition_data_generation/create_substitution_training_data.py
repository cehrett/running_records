"""
Takes a .csv file containing the the word "SUB" everywhere there is a word
that should be replaced. It then outputs that new text data.
"""

import pandas
from typing import List
import numpy as np
import logging
from tqdm import tqdm

tqdm.pandas()

def load_file(filename):
    """
    Loads the .csv file and returns a pandas dataframe.
    """
    return pandas.read_csv(filename)

def get_list_of_all_words(df: pandas.DataFrame) -> List[str]:
    """
    This should iterate over all of the sentences in our "original_text" column
    and return me a list of all unique words in the dataset.
    """
    all_words = set()

    for index, row in df.iterrows():
        for word in row["original_text"].lower().split():
            if word.isalpha():
                all_words.add(word)
    
    return list(all_words)
    

def get_new_word(mutated_sentence, old_sentence) -> str:
    return_word = np.random.choice(all_words)
    return return_word

def get_substituted_sentence(row):
    """
    When this sentence is called, it will return a sentence with the SUB
    replaced by a word that wasn't originally there.
    """
    original_text = row["original_text"]
    mutated_text = row["mutated_text"]

    while "SUB" in mutated_text:
        mutated_text = mutated_text.replace("SUB", get_new_word(mutated_text, original_text), 1)
    
    row["mutated_text"] = mutated_text

    return row

def main():
    """
    Main function.
    """
    # Load the .csv file
    logging.info("Loading the .csv file")
    data = load_file("repetition_data_generation/data/output_substitutions.csv")
    logging.info("Loaded the .csv file")

    global all_words
    all_words = get_list_of_all_words(data)

    # Apply the get_substituted_sentence function to each row in the dataframe
    logging.info("Applying get_substituted_sentence function to each row in the dataframe")
    data = data.progress_apply(get_substituted_sentence, axis=1)

    logging.info("Saving the .csv file")
    data.to_csv("repetition_data_generation/data/sub_data.csv", index=False)

if __name__ == "__main__":
    main()