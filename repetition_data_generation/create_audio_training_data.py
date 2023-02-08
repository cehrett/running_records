import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)

from tts_data_generation import process_and_save
from nemo_asr_transcript_generation import get_asr_transcript
import argparse
import pandas as pd
from tqdm import tqdm
import logging

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(
        description="Generate training data for TTS")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)

    args = parser.parse_args()
    mutated_df = pd.read_csv(args.input_file)
    output_filenames = []

    audio_paths = []
    transcripts = pd.DataFrame(columns=["audio_path", "asr_transcript"])

    num_rows_processed = 0
    # Iterate over each row in mutated_df
    print("Getting Audio Files...")
    for index, row in tqdm(mutated_df.iterrows(), total=mutated_df.shape[0], desc="Generating Audio Files"):
        if num_rows_processed >= args.num_samples:
            logging.info("Breaking out of loop. Reached max number of samples.")
            break
        if pd.isnull(row["mutated_text"]):
            logging.error("Skipping Problematic Row")
            continue
        process_and_save(row["mutated_text"], args.output_file + "_" + str(index) + ".wav")
        audio_paths.append(args.output_file + "_" + str(index) + ".wav")
        num_rows_processed += 1

    transcripts["audio_path"] = audio_paths
    transcripts["asr_transcript"] = get_asr_transcript(
        transcripts["audio_path"].to_list())

    transcripts.reset_index(inplace=True, drop=True)
    new_mutated_df = transcripts.join(mutated_df)
    new_mutated_df.to_csv(args.output_file + ".csv", index=False, mode='a')

    mutated_df.drop(mutated_df.index, inplace=True)
    mutated_df.to_csv(args.input_file, index=False)
