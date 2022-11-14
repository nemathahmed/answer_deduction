---
layout: default
---

<!-- ![Banner](assets/biscuit.png)


**[Biscuit](http://sblisesivdin.github.io/biscuit)** is a single-page responsive Jekyll theme. This is the most simple and still-good-looking Jekyll theme that you can find.  -->


<h1 style="text-align: center;"><Strong>Reliable Answer Deduction (RAD)</Strong></h1>
<h2 style="text-align: center; margin-top:-2.5rem;">Know When To And When Not To</h2>

<h3 style="text-align: center;">CS7641 | Semester Project | Group 33</h3>

[<img src="https://s18955.pcdn.co/wp-content/uploads/2018/02/github.png" width="25"/>](https://github.com/user/repository/subscription)

## Introduction

There is a plethora of textual information out there and it has become increasingly difficult to draw insights from the data and find the relevant answers to our questions. Question-answering systems like Machine Reading Comprehension (MRC) systems are effective in retrieving useful information, wherein the model retrieves the answer from given comprehensions instead of the web. A lot of attention has been given to MRC tasks lately. Although existing models achieve reasonably good results, they may still produce unreliable answers when the questions are unanswerable and are computationally heavy. Thus, our aim here is to experiment and present a model which is more reliable.


## Problem Definition

Our aim here is to leverage the power of Machine Learning and Natural Language Processing to create a model that deducts the answer when given a passage and also identifies if a question is unanswerable. We plan to develop an ensemble model which successfully accomplishes the task and gives reliable answers to the questions asked from the comprehension. Our proposed approach will be to innovate different modules in our architecture by taking inspiration from state of the art architectures. 


## Dataset
We are going to use Stanford Question Answering Dataset 2.0 ([SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/))  combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions. It was written adversarially by crowdworkers such that unanswerable questions look similar to answerable ones. To do well on SQuAD2.0, systems should answer questions when possible and also determine when no answer is supported by the paragraph and abstain from answering. 


A sample of raw dataset has been shown below showcasing questions with their actual or plausible answers. The *“is_impossible”* flag is provided to distinguish between answerable and unanswerable questions and the features for the question vary accordingly. 

## Sample Dataset Q&A format

![Sample Dataset](assets/q-a.jpeg "Sample Model Output")
 
## Sample Model Output ([Source](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/Normans.html?model=nlnet%20(single%20model)%20(Microsoft%20Research%20Asia)&version=v2.0)):

![Sample Model Output](assets/q_a.jpeg "Sample Model Output")



## Algorithms/Methods:
We would be using a Deep Learning architecture for our Machine Reading Comprehension (MRC) task. It would involve the following sections/tasks -
Embedding module, Feature extraction, Context question interaction, Verification module and answer prediction. 
We would use contextual embeddings from BERT and then experiment with the feature extraction techniques in combination with the attentive context question interaction methods. Span Extractor has been proven to work well as an answer predictor in MRC tasks in existing literature<sup>[7]</sup>  but we would be experimenting with other methods as well. We would also be exploring unsupervised models which learn via self-supervision.


## Potential results and Discussion
We hope to achieve competent scores on the popularly used metrics for this task which are F1 score and EM score. These scores are already used in the SQuaD<sup>[5]</sup> to compare various models on the dataset. Additionally, if we are able to develop a competent model, we would also like to focus on keeping the model light in terms of the model size, so that it could be deployed in places where computational resources are limited.

<!-- ## Data Exploration
![Answer](assets/Answer.png "Answer Length")
![Question](assets/Question.png "Question Length")
![Context](assets/Context.png "Context Length")
![QContext](assets/Q_context.png "Context Length")
<!-- ![Title](assets/Title.png "Context Length") -->

<!-- ![Start From Fraction]('assets/Start From Fraction.png' "SF Length") --> -->



## Gantt Chart
![Gantt Chart](assets/gantt.jpeg "Gantt Chart")

# Midterm

### Dataset Preprocessing & Exploration

When it comes to SQUAD 2.0 dataset, we found it to be mostly reliable and clean. However, we did do some basic cleaning exercise of removing the additional white spaces, conversion to lower case, stripping of unknown ASCII characters and tokenization as per the model requirement. Datapoints which had unreasonably small question length have been removed from both Training and Testing datasets. In order to convert the words in the passage to their root form to be in sync with the answers, we have used Lemmatization technique. Feature engineering was also done to find the end character of the answers given we have been provided with the start character.

The training dataset of SQUAD 2.0 is unbalanced with two thirds of questions being "Answerable" and the testing dataset is highly balanced with the "Answerable" questions comprising 49.9% of the data as can be seen below.

<table>
  <tr>
    <td><img src="assets/train_ques.png" width="400"/></td>
    <td><img src="assets/test_ques.png" width="400"/></td>
  </tr>
 </table>

We also did an analysis to understand the distribution of questions per passage across the train and test datasets and concluded that test dataset has an average of 10 questions being asked per context passage and the distributions are shown below:

<table>
  <tr>
    <td><img src="assets/Ques_pp_pt_Train.png" width="400"/></td>
    <td><img src="assets/Ques_pp_pt_Test.png" width="400"/></td>
  </tr>
 </table>

Next, we looked at the lengths of the context, questions and answers to understand the dataset better and identify any outliers

<table>
  <tr>
    <td><img src="assets/Context.png" width="400"/></td>
    <td><img src="assets/Question.png" width="400"/></td>
   <td><img src="assets/Answer.png" width="400"/></td>
  </tr>
 </table>
 
Lastly, we looked at the dominant words in the contexts and the questions by creating a wordcloud after removing the stopwords
 
 <table>
  <tr align="middle">
    <td>Wordcloud for given Contexts</td>
    <td>Wordcloud for given Questions</td>
  </tr>
  <tr>
    <td><img src="assets/wordcloud_context.png" width="400"/></td>
    <td><img src="assets/wordcloud_question.png" width="400"/></td>
  </tr>
 </table>

## UnSupervised Learning

### Generative Pre-trained Transformer 3

<p float="left" align="middle">
 <img src="assets/xx.jpeg" width="700"/>
</p>

GPT-3 is an autoregressive language model which shows strong few shot learning capability on natural language tasks. It has a transformer architecture which is trained using the generative pre-training method. It has the capability to produce human-like answers. Since it can perform new tasks that it hasn’t been trained on, it can be used as an unsupervised method. Hence, we use GPT-3 with engine ‘text-davinci-002’ for our question answering task without providing the answers to it, more specifically, we ask GPT-3 to perform the following tasks to generate an answer. For the first task, we provide the model with only the question and task prompt `Give an answer of length less than x words.`. For the second task, we provide the model with the question, the context paragraph and the task prompt - `Based on the context below give an answer to the question below. The answer should have less than x words`. For both the tasks we provide GPT-3 with an answer length based on the dataset answer length to avoid undesirably long answers. On each answer we obtain the similarity by comparing the embeddings of GPT answer and the SQuAD dataset with the 'text-similarity-davinci-001' engine as shown in the image shown below. Finally, we compare the answers generated by the model with the answer provided in the dataset by computing L2 similarity score.

<p float="left" align="middle">
 <img src="https://cdn.openai.com/embeddings/draft-20220124e/vectors-1.svg" width="700"/>
</p>


### GPT-3 Output Analysis:

Under the unsupervised method using GPT-3 we analyzed the statistics on the answers. We compare GPT-3’s performance with the SQuAD dataset.  We use a subset of 500 datapoints and use this to derive insights.

As discussed we assign GPT-3 the task of giving answers to our questions. We do it in branches. Firstly, we don’t give GPT-3 the context to our question. We allow it to use it’s vast knowledge gained by being trained on millions of articles to answer a question. Based on the word length of answers in SQuAD we try asking GPT-3 to limit it’s answer length to avoid unnecessary information leading to score diversion.

In the plot below, we see that our dataset gives much more concise answers compared to the GPT model. Also we see roughly the GPT answers are much more verbose. This comes from the fact that GPT has knowledge of things which are not necessarily in the context used in the SQuAD dataset. GPT roughly gives an answer of character length of 41 on average. 
<table>
  <tr>
    <td><img src="assets/gpt_raw_ans_length (1).png" width="400"/></td>
  </tr>
 </table>


Now checking the similarity between answers from GPT-3 and SQuAD dataset we get a similarity plot as shown below.

We see that the graph is not skewed towards score of 1 meaning the fraction where GPT-3 results overlap with SQuAD considerably is less. This is probably due to the additional information given by GPT3. The average similarity score or accuracy is 81.9%.
```
Example (No Context): 
* Question: When did Beyonce start becoming popular?
* GPT-3 Answer: Beyonce started becoming popular in the early 2000s.
* SQuAD Answer: In the late 1990s.
```

<table>
  <tr>
    <td><img src="assets/gpt_ans_length.png" width="400"/></td>
  </tr>
 </table>

Now when we give GPT the context along with the question we see interesting insights being derived. Firstly, we that the answer length now starts matching the answers from SQuAD dataset. And there is no divergence in the answer length between SQuAD and GPT-3. This is much more closer to SQuAD dataset. The average answer length is 28.0.
<table>
  <tr>
    <td><img src="assets/similarity_raw.png" width="400"/></td>
  </tr>
 </table>

As expected we also see a huge improvement in the similarity scores between GPT-3 answers and SQuAD answers. We see that many questions are now answered with 100% accuracy. The average accuracy is 86.4%. Thus providing context makes answers both concise and precise to SQuAD.

```
Example (With Context): 
* Question: When did Beyonce start becoming popular?
* GPT-3 Answer: In late 1990s.
* SQuAD Answer: In the late 1990s.
```

<table>
  <tr>
    <td><img src="assets/similarity.png" width="400"/></td>
  </tr>
 </table>


We also verify the plausible answers to which SQuAD thinks it’s improbable to answer the question. From the graph below we see that when we give the context - the questions which are termed unanswerable and have a plausible answer are similar to GPT-3 predictions. On the other hand there is divergence when we don’t give the context. This is because without context GPT-3 takes in all the knowledge and answers all questions. 

<table>
  <tr>
    <td><img src="assets/imp_similarity.png" width="400"/></td>
  </tr>
 </table>

Scores for the current fine tuned model:
<table align="middle">
  <tr>
    <td>Measure</td>
    <td>Context</td>
    <td>Without Context</td>
  </tr>
  <tr>
    <td>Similarity Avg Score (%)</td>
    <td>86.4</td>
    <td>81.9</td>
  </tr>
  <tr>
    <td>Average Answer Length</td>
    <td>28.0</td>
    <td>41.0</td>
  </tr>
  <tr>
    <td>Max Answer Length</td>
    <td>251</td>
    <td>306</td>
  </tr>
  <tr>
    <td>Plausible Answer Similarity (%)</td>
    <td>82.4</td>
    <td>82.0</td>
  </tr>
 </table>
 
Now we analyze the answers generated by CPT across the various commonly asked questions like "Who", "What", "How", and "Why" against the actual answers provided by the SQuAD 2.0 dataset.

 <table>
  <tr>
    <td><img src="assets/Answer_who_gpt.png" width="400"/></td>
    <td><img src="assets/Answer_what_gpt.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="assets/Answer_how_gpt.png" width="400"/></td>
    <td><img src="assets/Answer_why_gpt.png" width="400"/></td>
  </tr>
 </table>
 
 


## Supervised Learning

### Bi-Directional Attention Flow (BiDAF) model: 
	
Bi-Directional Attention Flow (BiDAF) is a multistage network which uses bi directional attention flow mechanism to model a query-aware context representation. We use this architecture for our question answering task due it’s effective attention computation at every time step which reduces information loss. Additionally, since the attention computed at every time step is a function of the context paragraph and the question at the current time step, it allows the model to have a memory-less attention mechanism which enables the model to learn the interaction between the given context and the question.

The BiDAF architecture details can be found here [Source<sup>[8]</sup>]:
<table>
  <tr>
    <td><img src="assets/Arch.jpeg" width="400"/></td>
  </tr>
 </table>
 
 
Below are the results and training details for our preliminary fine tuning:
<table>
  <tr align="middle">
    <td>Train/NLL</td>
    <td>Dev/NLL</td>
  </tr>
  <tr>
    <td><img src="assets/train_NLL (1).svg" width="400"/></td>
    <td><img src="assets/dev_NLL.svg" width="400"/></td>
  </tr>
 </table>
 
Scores for the current fine tuned model:
<table align="middle">
  <tr>
    <td>F1 Score</td>
    <td>54.91</td>
  </tr>
  <tr>
    <td>EM Score</td>
    <td>51.79</td>
  </tr>
  <tr>
    <td>AnVA Score</td>
    <td>62.04</td>
  </tr>
 </table>
 
### Analysis of fine tuning:
The training is highly sensitive to the following parameters:
* Batch size,
* Learning rate 
* Max answer length

The plots above are for batch size = 64, learning rate =  0.5 and max answer length = 15. 

The calculations above are optimally computed using a few experiments until now but these are subject to change once we experiment with the parameters exhaustively. The model is still low relative to its potential on F1 and EM scores of BiDAF models tuned on this dataset. An interesting point to note is that the model has performed relatively better on AnVA metric which basically measures the classification accuracy of the model when only considering its answer (any span predicted) vs. no-answer predictions. This is due to the architecture of BiDAF which allows it to compare the predicted answer versus the no answer hypothesis effectively. One major challenge that we have faced is the limited availability of computing resources. Training process took a lot of time computationally and made effective testing with more combinations of hyperparameter tuning completely infeasible. We have tried reducing the training dataset points to deal with the issue, but it leads to higher loss on dev validation sets as well. We are yet to reach an estimate for the optimal point of this tradeoff.

## Further Steps
We would further tune the BiDAF model experimenting with more combinations of hyperparameters. Going ahead we would be experimenting with relevant BERT based models for the QA task and performing comparative studies for the fine tuned models.





## References:

1. [Retrospective Reader for Machine Reading Comprehension](https://arxiv.org/abs/2001.09694) 
2. [NEURQURI: NEURAL QUESTION REQUIREMENT INSPECTOR FOR ANSWERABILITY PREDICTION IN MACHINE READING COMPREHENSION](https://openreview.net/pdf?id=ryxgsCVYPr)
3. [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822) 
4. [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/pdf/2004.02984)
5. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf)
6. [Bridging the Gap between Language Model and Reading Comprehension: Unsupervised MRC via Self-Supervision](https://arxiv.org/pdf/2107.08582v1.pdf)
7. [Neural Machine Reading Comprehension: Methods and Trends](https://arxiv.org/abs/1907.01118) 
8. [BI-DIRECTIONAL ATTENTION FLOW FOR MACHINE COMPREHENSION](https://arxiv.org/pdf/1611.01603.pdf)


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
