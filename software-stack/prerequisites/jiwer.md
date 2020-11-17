---
description: Similarity measures for automatic speech recognition evaluation
---

# JiWER

## Overview

JiWER is a python package that can be used to calculate the Word Error Rate \(WER\) of an automatic speech recognition system. In addition approximating the WER, it can also approximate the Match Error Rate \(MER\), Word Information Lost \(WIL\) and Word Information Preserved \(WIP\) of a transcript. For this research project, we will be using JiWER to evaluate our Watson Speech-to-Text model and determine its overall effectiveness, as well as its demographic blind spots if any exist.

{% page-ref page="../../testing/model-evaluation.md" %}

### Example Case

```python
import jiwer

ground_truth = "i like monthy python what do you mean african or european swallow"
hypothesis = "i like python what you mean or swallow"

error = jiwer.wer(ground_truth, hypothesis)
```

In this example, we use jiwer to calculate the WER between the ground truth sentence and the hypothesis sentence. We will be applying a similar methodology for our model evaluation.

## Pypi Page

For more information on JiWER, please click on[ this link](https://pypi.org/project/jiwer/) to view the original documentation PyPI.



