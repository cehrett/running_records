# This package is intended to develop a script that will take as input a
# running record audio file and ground truth transcript, and output a
# transcript of the audio along with metadata that constitute a scoring
# of the running record.

from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pandas as pd
import numpy as np
import fuzzy
import string

# First define helper functions.

# input string, get metaphone representations of each word in the string
def get_dmeta(string):
  """
  Get double metaphone representation of a string. Written by Dillon Ranwala
  """
  dmeta=fuzzy.DMetaphone()
  splitstring = string.split()
  stringlist = []
  encoding = 'utf-8'
  for word in splitstring:
    stringlist.append(dmeta(word)[0])
    
  # Converts nonetypes to bytes for empty string
  stringlist = [(bytes('',encoding)) if word is None else word for word in stringlist]
  
  #decoding bytes into a unicode string for each word
  bytes2str = []
  for byte in stringlist:
    b2str = byte.decode(encoding)
    bytes2str.append(b2str)

  finalstr = ' '.join(bytes2str)
  return finalstr

def levenshtein_distance_matrix(string1, string2, is_damerau=True):
  n1 = len(string1)
  n2 = len(string2)
  d = np.zeros((n1 + 1, n2 + 1), dtype=int)
  for i in range(n1 + 1):
      d[i, 0] = i
  for j in range(n2 + 1):
      d[0, j] = j
  for i in range(n1):
      for j in range(n2):
          if string1[i] == string2[j]:
              cost = 0
          else:
              cost = 1
          d[i+1, j+1] = min(d[i, j+1] + 1, # insert
                            d[i+1, j] + 1, # delete
                            d[i, j] + cost) # replace
          if is_damerau:
              if i > 0 and j > 0 and string1[i] == string2[j-1] and string1[i-1] == string2[j]:
                  d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + cost) # transpose
  return d

def location_tag(truth: str, hypothesis: str, is_damerau=True, mark_deletions = False):
  """
  Returns an array the same length as hypothesis, where the ith entry is an
  integer value giving the index of the location of the ith element of
  hypothesis in truth. Elements of hypothesis that don't match truth are given
  value -1 if insertion, -2 if substition.
  Optionally, deletions can be marked as well, but this means the resulting
  output length won't match the hypothesis length.

  Examples: for truth='kits' and hypothesis='kites', output=[0,1,2,-1,3].
            for truth='kits' and hypothesis='mites', output=[-2,1,2,-1,3].
            for truth='kits' and hypothesis='mis', output=[-2,1,3].
  This code builds on code found at
  https://gist.github.com/jlherren/d97839b1276b9bd7faa930f74711a4b6.
  """
  dist_matrix = levenshtein_distance_matrix(truth, hypothesis, is_damerau)
  i, j = dist_matrix.shape
  i -= 1
  j -= 1
  ops = list()
  while i != -1 and j != -1:
      if is_damerau:
          if i > 1 and j > 1 and truth[i-1] == hypothesis[j-2] and truth[i-2] == hypothesis[j-1]:
              if dist_matrix[i-2, j-2] < dist_matrix[i, j]:
                  # ops.insert(0, ('transpose', i - 1, i - 2))
                  ops.insert(0, (i-2))
                  ops.insert(0, (i-1))
                  i -= 2
                  j -= 2
                  continue
      index = np.argmin([dist_matrix[i-1, j-1], dist_matrix[i, j-1], dist_matrix[i-1, j]])
      if index == 0:
          if dist_matrix[i, j] > dist_matrix[i-1, j-1]:
              ops.insert(0, -2)
          elif i>0:
              ops.insert(0, (i-1))
          i -= 1
          j -= 1
      elif index == 1:
          ops.insert(0, -1)
          j -= 1
      elif index == 2:
          if i+j>0:
            if mark_deletions:
              ops.insert(0, -3)
          i -= 1
  return ops

# Now define the class object
class Running_record:
    """
    This class scores a running record audio file using a ground truth transcript.
    TODO: Figure out how to call a pre-trained WSTT model for transcription
    """

    def __init__(self, audio_filepath: str, wstt_credentials: dict, ground_truth: str):
        self.audio_filepath = audio_filepath
        self.wstt_credentials = wstt_credentials
        self.ground_truth = ground_truth

    def process_running_record(self):
        """
        Call the class functions necessary to process the running record. I.e., get
        a WSTT transcript, convert that transcript to a pandas df, add a phonetic
        representation to that df, and then score the resulting df as a running
        record using the ground truth transcript.
        """
        self.get_WSTT_transcript(self.audio_filepath, self.wstt_credentials)
        self.convert_json_to_pddf()
        self.convert_string_to_pddf()
        self.add_phonetic_representation()
        self.score_transcript()

    def get_WSTT_transcript(self,
                            audio_filepath=None,
                            wstt_credentials=None):
        """
        Use the Watson Speech to Text (WSTT) credentials stored in wstt_credentials to
        get a WSTT transcript of the audio file stored at audio_filepath.

        Inputs
        ----------
        audio_filepath: str
          filepath of audio
        wstt_credentials: dict
          dict containing the following:
            authenticator -- authentification code for WSTT instance
            acoustic_id   -- acoustic model customization id
            language_id   -- language model customization id

        Returns
        ----------
        transcript_json: str
          json format

        It is recommended to use a customized, pretrained instance of WSTT.
        """
        # Define inputs
        if audio_filepath == None: audio_filepath = self.audio_filepath
        if wstt_credentials == None: wstt_credentials = self.wstt_credentials

        # Authenticate into the Speech to Text service
        authenticator = IAMAuthenticator(wstt_credentials['authenticator'])
        speech_to_text = SpeechToTextV1(
            authenticator=authenticator
        )
        speech_to_text.set_service_url('https://stream.watsonplatform.net/speech-to-text/api')

        with open(audio_filepath, 'rb') as audio_file:
            transcript_json = speech_to_text.recognize(
                audio=audio_file,
                acoustic_customization_id=wstt_credentials['acoustic_id'],
                content_type='audio/mp3',
                word_alternatives_threshold=0,
                speaker_labels=False
            ).get_result()

        self.transcript_json = transcript_json

    def convert_json_to_pddf(self, transcript_json=None):
        """
        Take a JSON string output by WSTT and convert it into a pandas DF.

        Inputs
        ----------
        transcript_json: str
          json format, output from WSTT
          Default: self.transcript_json

        Returns
        ----------
        transcript_df: pd.DataFrame
          A dataframe with the following columns:
          ==========  ==============================================================
          word_index  (ordinal that tells which word in the the transcript this is.
                       E.g., if the transcript gives two alternatives for the same
                       spoken word, there will be two rows sharing the same
                       word_index.)
          word        (self-explanatory)
          confidence  (WSTT returns confidence values for its multiple alternatives)
          start_time  (WSTT returns the time index for the beginning and end of a word)
          end_time    (WSTT returns the time index for the end of a word)
          ==========  ==============================================================
        """

        # Define inputs
        if transcript_json == None: transcript_json = self.transcript_json

        # Initialize column names of output DataFrame, and useful variables
        names = ['word_index', 'word', 'confidence', 'start_time', 'end_time']
        word_index = 0
        data = []

        # Loop through the results, words, and word alternatives in the json
        # Collect the info we want in a list which will later become a pd.DataFrame
        for result in transcript_json['results']:
            for word in result['word_alternatives']:
                for alternative in word['alternatives']:
                    entries = [word_index,
                               alternative['word'],
                               alternative['confidence'],
                               word['start_time'],
                               word['end_time']]
                    data.append(dict(zip(names, entries)))
                word_index += 1

        # Convert to pd.DataFrame and save
        transcript_df = pd.DataFrame(data)
        self.transcript_df = transcript_df

    def convert_string_to_pddf(self, ground_truth=None):
        """
        Take as input a string (the ground truth transcript) and convert it to a
        pandas data frame with a column corresponding to location in the text (e.g.
        first word has value 0), and column containing the words of the text.

        Inputs
        ----------
        ground_truth: str
          String containing the ground truth transcript.
          Default: self.ground_truth

        Returns
        ----------
        ground_truth_df: pd.DataFrame
          A dataframe with the following columns:
          ==========  ==============================================================
          word_index  (ordinal that tells which word in the the transcript this is.
          word        (self-explanatory)
          ==========  ==============================================================
        """

        # Define inputs
        if ground_truth == None: ground_truth = self.ground_truth

        # Convert to lower, remove punctuation
        ground_truth = ground_truth.lower()
        exclude = set(string.punctuation)
        ground_truth = ''.join(ch for ch in ground_truth if ch not in exclude)

        # Make into DataFrame
        ground_truth_df = pd.DataFrame(ground_truth.split(' '), columns=['word'])
        # ground_truth_df.rename(columns={"0": 'word'},inplace=True)

        self.ground_truth_df = ground_truth_df

    def add_phonetic_representation(self,
                                    transcript_df=None,
                                    ground_truth_df=None):
        """
        Take as input a pandas DF containing a column "word", and output the same DF
        with a new column added: "phonetic", which is a phonetic representation of
        "word".

        Inputs
        ----------
        transcript_df: pd.DataFrame
          A dataframe containing the column "word"
          Default: self.transcript_df

        Returns
        ----------
        transcript_df: pd.DataFrame
          A dataframe containing columns "word" and "phonetic".
        """

        # Define inputs
        if transcript_df == None: transcript_df = self.transcript_df
        if ground_truth_df == None: ground_truth_df = self.ground_truth_df

        # Add phoneticization
        transcript_df['phonetic'] = transcript_df[['word']].apply(lambda x: get_dmeta(x[0]), axis=1)
        ground_truth_df['phonetic'] = ground_truth_df[['word']].apply(lambda x: get_dmeta(x[0]), axis=1)

        # Update stored transcripts
        self.transcript_df = transcript_df
        self.ground_truth_df = ground_truth_df

    def score_transcript(self,
                         transcript_df=None,
                         ground_truth_df=None,
                         is_damerau=True):
        """
        Take as input two dataframes each with column "phonetic", and use one of
        them as the ground truth with which to produce a running record score of the
        first.

        Inputs
        ----------
        transcript_df: pd.DataFrame
          DataFrame with column "phonetic"; treated as hypothesis.
          Default: self.transcript_df
        self.ground_truth_df: pd.DataFrame
          DataFrame with column "phonetic"; treated as ground truth

        Returns
        ----------
        transcript_df: pd.DataFrame
          transcript_df with new column "score", which encodes the status of the
          hypothesis row as either correct, substitution, insertion, or repetition.
          # TODO: implement repetition tag.
          # TODO: make compatible with multiple hypotheses for each word.
        """

        # Define inputs
        if transcript_df == None: transcript_df = self.transcript_df
        if ground_truth_df == None: ground_truth_df = self.ground_truth_df

        # Get location tags and adding a score
        score = location_tag(truth=ground_truth_df.phonetic,
                             hypothesis=transcript_df.phonetic,
                             is_damerau=is_damerau,
                             mark_deletions=False)
        transcript_df['score'] = score

        # Grouping together word alternatives when none are correct (leaves only the one with the highest confidence)
        transcript_df['wrong_tag'] = np.where(transcript_df['score'] < 0, True, False)
        transcript_df = transcript_df.groupby(['word_index', 'wrong_tag'], group_keys=False).apply(
            lambda x: x.loc[x.confidence.idxmax()])

        transcript_df = transcript_df.reset_index(level=1, drop=True)

        # Finds remaining word alternatives and labeling them with a temp column
        transcript_df['match'] = transcript_df.word_index.eq(transcript_df.word_index.shift())
        transcript_df['match2'] = transcript_df.word_index.eq(transcript_df.word_index.shift(-1))
        transcript_df['dup'] = transcript_df.match | transcript_df.match2

        # Removes any remaining word alternatives if one of them is correct
        transcript_df = transcript_df.loc[~((transcript_df['dup'] == True) & (transcript_df['score'] < 0))]

        transcript_df = transcript_df.drop(columns=['match', 'match2', 'dup', 'word_index', 'wrong_tag']).reset_index()

        self.transcript_df = transcript_df


if __name__=="__main__":
    # TODO write some tests to verify the package functionality
    print("RunningRecords imported.")
