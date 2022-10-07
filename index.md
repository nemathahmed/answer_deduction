---
layout: default
---

<!-- ![Banner](assets/biscuit.png)


**[Biscuit](http://sblisesivdin.github.io/biscuit)** is a single-page responsive Jekyll theme. This is the most simple and still-good-looking Jekyll theme that you can find.  -->


<h1 style="text-align: center;">Reliable Answer Deduction <br />  Know When To And When Not To</h1>

<h3 style="text-align: center;">CS7641 | Semester Project | Group 33</h3>

[<img src="https://s18955.pcdn.co/wp-content/uploads/2018/02/github.png" width="25"/>](https://github.com/user/repository/subscription)

## Introduction

There is a plethora of textual information out there and it has become extremely crucial to identify whether the given information is important. To find relevant information time is spent on going through the passage and finding the answer. However, plenty of times it has been seen that extractive reading comprehension systems locate the correct answer to a question, but they also tend to make unreliable guesses on questions for which the correct answer is not stated in the context, which is far from true language understanding.

## Problem Definition

Our aim here is to leverage the power of machine learning and natural language processing to create a model that deducts the answer when given a passage and also identifies if a question is unanswerable. We plan to develop an ensemble model which successfully accomplishes the task and gives reliable answers for the questions asked from the comprehension. Our proposed approach will be to use a combination of architectures from the state of the art models which are fairly accurate.  



## Dataset
We are going to use Stanford Question Answering Dataset 2.0 ([SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)) combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions. It was written adversarially by crowdworkers such that unanswerable questions look similar to answerable ones. To do well on SQuAD2.0, systems should answer questions when possible and also determine when no answer is supported by the paragraph and abstain from answering. 


Such a sample dataset has been shown below showcasing questions with their actual or plausible answers. “is_impossible” flag has been used to distinguish between answerable and unanswerable questions and the features for the question vary accordingly. We also have a “context” feature associated with a set of questions which contains the relevant passage. 


## Sample Dataset Q&A format

```json
 {
    "question": "What century did the Normans first gain their separate identity?",
    "id": "56ddde6b9a695914005b962c",
    "answers": [{
        "text": "10th century",
        "answer_start": 671
    }, {
        "text": "the first half of the 10th century",
        "answer_start": 649
    }, {
        "text": "10th",
        "answer_start": 671
    }, {
        "text": "10th",
        "answer_start": 671
    }],
    "is_impossible": false
}, {
    "plausible_answers": [{
        "text": "Normans",
        "answer_start": 4
    }],
    "question": "Who gave their name to Normandy in the 1000's and 1100's",
    "id": "5ad39d53604f3c001a3fe8d1",
    "answers": [],
    "is_impossible": true
}

```
## Sample Model Output ([Source](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/Normans.html?model=nlnet%20(single%20model)%20(Microsoft%20Research%20Asia)&version=v2.0)):

![Sample Model Output](assets/q_a.jpeg "Sample Model Output")



## Algorithms/Methods:
We would be using a deep learning architecture for our Machine Reading Comprehension(MRC) task. It would involve the following sections/tasks -
Embedding module, Feature extraction, Context question interaction, verification module and answer prediction. 
We would use contextual embeddings from BERT and then experiment with the feature extraction techniques in combination with the attentive context question interaction methods. Span Extractor has been proven to work well as an answer predictor in MRC tasks in existing literature[7] but we would be experimenting with other methods as well.

## Potential results and Discussion
We hope to achieve competent scores on the popularly used metrics for this task which are F1 score and EM score. These scores are already used in the SQuaD[5] to compare various models on the dataset. Additionally, if we are able to develop a competent model, we would also like to focus on keeping the model light in terms of the model size, so that it could be deployed in places where computational resources are limited.

## Gantt Chart
![Gantt Chart](assets/gantt.jpeg "Gantt Chart")

## References:

1. [Retrospective Reader for Machine Reading Comprehension](https://arxiv.org/abs/2001.09694) 
2. [NEURQURI: NEURAL QUESTION REQUIREMENT INSPECTOR FOR ANSWERABILITY PREDICTION IN MACHINE READING COMPREHENSION](https://openreview.net/pdf?id=ryxgsCVYPr)
3. [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822) 
4. [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/pdf/2004.02984)
5. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf)
6. [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822) 
7. [Neural Machine Reading Comprehension: Methods and Trends](https://arxiv.org/abs/1907.01118) 


## Team


* [Harsh Goyal](mailto:hgoyal34@gatech.edu)
* [Kritika Venkatachalam](mailto:kvenkata8@gatech.edu)
* [Nemath Ahmed](mailto:nshaik6@gatech.edu) 
* [Pankhuri Singh](mailto:psingh374@gatech.edu)
* [Siddharth Singh Solanki](mailto:siddharth.solanki@gatech.edu)

<!-- ### Files

* `_config.yml`            : Main configuration file.
* `index.md`               : Website page (for now, this page).
* `_includes/head.html`    : File to add custom code to `<head>` section.
* `_includes/scripts.html` : File to add custom code before `</body>`. You can change footer at here.
* `_sass` folder           : Related scss files can be found at this folder.
* `css/main.csss`          : Main scss file.
* `README.md`              : A simple readme file.

## Example tag usage

## Header 1
### Header 2
#### Header 3
**bold**
*italic*

> blockquotes

~~~python
import os,time
print ("Biscuit")
~~~

## Licence and Author Information

Biscuit is derived from currently deprecated theme [Solo](http://github.com/chibicode/solo). The development of Biscuit is maintained by [Sefer Bora Lisesivdin](https://lrgresearch.org/bora).

Biscuit and the previous code where Biscuit is derived are distributed with [MIT license](https://github.com/sblisesivdin/biscuit/blob/gh-pages/LICENSE).
 -->
