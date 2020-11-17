# Model Evaluation

## Word Error Rate

In order to evaluate our model, we are going to calculate the Word Error Rate of each of the model-generated transcripts for each piece of training audio. In order to give the model its best chance of success, we will be evaluating all of the hypotheses the model provides for each audio. As a result, of this methodology, we will not be constrained to using the transcript the model felt most confident in, and can instead see if any of its possible interpretations are accurate. We will also be taking note of the Word Error Rate of the model's most confident hypothesis, as this would preferably be the one we would use in production of this model.

### Word Error Rate Formula

The formula for Word Error Rate is listed below as the total number of substitutions, deletions and insertions divided by the number of words in the ground truth.

$$
WER = (S + D + I) / N
$$

### Results

TBD

