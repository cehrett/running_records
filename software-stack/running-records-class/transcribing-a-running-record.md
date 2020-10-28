# Transcribing a Running Record with Watson Speech-To-Text

## Watson's Involvement

For this project, we will be using IBM Cloud's Watson Speech-to-Text service in order to transcribe the child's audio into actual text that can be compared to a ground truth.

### Training the Model

In order to transcribe the audio recordings of the children reading audio, each of our research team members transcribed 7-9 recordings of children completing a Running Record exam. Each of these exams came from a diverse range of schools in South Carolina, with all of them being tested against the text **Old Man Moss**. By transcribing these audio recordings by hand, we were able to feed these recordings into a custom Watson Speech-To-Text Model that will be trained on the children's readings. As a result, our hope is that we can develop a baseline in seeing if Watson can accurately transcribe speech to text.

#### Note on Hesitations

When Watson transcribes audio, it will also detect when there is a hesitation in the speaker's voice. In order to compensate for this, each training audio has been trained to include approximate locations of the hesitations.

### Model Evaluation

Further information will soon be reported based on the effectiveness of our model. Currently, we plan on checking to see if our model has any demographic blind spots, and our hope is to minimize those blind spots to provide a more fair model.

## Implementing the Model

With the model now trained, we can now implement our model and transcribe new recordings of the child's audio. This can be done by passing audio files to the custom-trained model.





