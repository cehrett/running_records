---
description: Introduction to the IBM Cloud tools that we will be using for this project.
---

# Introduction to IBM Cloud

## Overview

The IBM Cloud is a collection of tools that we can use for creating various Machine Learning models. For this research project, we will be taking advantage of Watson Speech to Text Model, which can be used to convert audio files into transcripts. Normal applications of this service include powering automated services such as voice agents. In the context of scoring Running Records, this is a great starting point for our goal of automating the scoring of a Running Record, as we will need to produce a textual version of the child's speech.

### Hesitations and &lt;eps&gt;

When translating speech to text, sometimes the model will denote certain parts of speech as hesitations and &lt;eps&gt;. A hesitation occurs when the speaker stutters when introducing a word. These events are denoted with a **%HESITATION** in the transcript. Meanwhile, &lt;eps&gt; are a little more mysterious in nature. While there is no official documentation on the meaning of this symbol, it seems to correspond with some speech that is not silent, but at the same time not an actual English word. This is the definition we have settled upon through our own testing, and when we have a more concrete answer, we will make sure to update this section of the documentation.

### Documentation

For further documentation on Watson Speech-to-Text,[ click here](https://cloud.ibm.com/apidocs/speech-to-text) to view IBM's documentation on the tool. 

