# Data Generation

This folder is home to our scripts that we use for generating data. Here, we can
pass in an input file (currently the minutes of the European Parliament) and
generate data containing mutated sentences. These mutated sentences are then fed
into TTS and ASR models that generate the audio data.

## Running on the Palmetto Cluster

### The First Time

The first time you run this, you will need to log into the cluster and do a few
things. This can be done in a Jupyter Notebook instance.

1. Clone this repository to the cluster.

2. Create a virtual environment in the root of the repository. This will need to
   be called `venv`. To do this, run `python -m venv venv` in the root of this
   repository.

3. Inside the repository, run `pip install -r requirements.txt` to install the
   required packages.

4. As a sanity check, run the `tts_data_generation.py` and `nemo_asr_data_generation.py`
   scripts. If these succesfully execute, we are good to go. If not, there may be
   dependencies not originally installed by pip you need to hunt down.

### General Execution Steps

If you want to run this on the Palmetto Cluster, here's what you need to do.

1. Go to the [Palmetto Cluster Job Dashboard](https://openod02.palmetto.clemson.edu/pun/sys/myjobs/)
   and select `new job.` From there, choose `default template` and it will create a new job,

2. Select the new job that was created and click on `open editor`. This will open the job submission
   script in the dashboard editor.

3. In the editor, copy and paste what is in `palmetto_job.pbs` into the editor. This script tells
   the cluster what resources we want to request, who to email and when, and the steps we
   want to execute. Once you are done, save it!

4. Navigate back to the dashboard and click submit! Now your job is running. You will get an email
   when the job completes, describing its exit scenario.
