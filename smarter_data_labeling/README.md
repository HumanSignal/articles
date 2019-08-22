# Introduction

The vast amount of deep learning tools enables us to quickly build new
apps with incredible performance, from computer vision classifying
complex objects on photo to natural language understanding by
extracting semantics from texts. However, the main bottleneck for all
these methods is the massive amount of data needed to train all these
models — typically hundred thousands of training examples.

If you are starting to build image classifier from scratch, say, for
detecting stale products on a lane, you’ll get stuck for weeks or
months to collect and manually annotate all these photos. Fortunately,
there are a bunch of deep neural networks, that are already
pre-trained on large image datasets with many classes, and ready to
use for mitigating that cold start problem. The core idea of transfer
learning is to leverage the outputs of these models, capturing
high-level image semantics, as inputs to new classifiers that solve
the target task. It significantly reduces the amount of data needed to
be annotated by human labelers, from hundred thousand to thousands of
images.

However, even annotating thousands of examples could be expensive,
especially if each annotation task relies on subject matter expert’s
knowledge. Ideally, it would be perfect for labeling only a few
hundreds of tasks and letting deep learning machinery do self-learning
without any supervision. This problem is also known as budgeted
learning: we economize the amount of money to be spent on acquiring
training dataset to build the model of required performance. Another
problem is related to a concept drift — when the target task is
changing over time (new products are coming to the detector’s line)
and prediction performance degrades without human intervention.

# Learning smarter

So, besides what transfer learning offers, can we further reduce the
amount of labeling work? Actually, the answer is yes, and there are a
couple of techniques that exist. One of the most well studied is
active learning. The principle is straightforward: only label what is
useful for your current model. Formally, the active learning algorithm
could be summarized in the following steps:

1. Train initial model
2. Pick one most useful sample (one that model is uncertain about,
   based on it’s predicted class probabilities)
3. Label this sample and add it to a training set
4. Retrain model on a new training set
5. Repeat from step 2

This heuristic works pretty well with old-school machine learning
models (e.g. linear classifiers like logistic regression, SVM, etc.),
but empirical studies suggest that it becomes non effective in deep
learning setting. Another rarely mentioned drawback is that classical
active learning requires retraining the model after each new labeled
sample, which is a time-consuming process.

So, summarizing, to enable a quick training of a new classifier
without any labeled data, with minimal human efforts, we have to deal
with the following:

1. Make use of the feature space produced by some high-quality
   pre-trained models
2. Select only the most informative samples to be labelled by human
   annotators
3. Quickly update our current machine learning state

The first step is very well studied in many research papers and
reviews, and even large enterprise companies share nowadays their
pre-trained neural networks, that could be very easily plugged into
feature extraction pipeline in the manner **vectorized_examples =
pretrained_model(raw_examples)**. The ideal starting point is to look at
TensorflowHub

Next steps are a tricky art of machine learning. Though intuition
behind the informativeness of the sample is clear, it’s unclear how to
calculate it accurately. Therefore there exist many different measures
based on class probabilities p₁(x), p₂(x), …, pᵢ(x),…, sorted in
descending order:

- Uncertainty: u(x) = 1 — p₁(x)
- Margin: m(x) = p₂(x) — p₁(x)
- Entropy: e(x) = \sum_{i=1}^K {p_i(x)*\log p_i(x)}

Which are all subjected to be maximized over seekable example set x ∈
X. Once we found such ambiguous example x, we hopefully see the model
be much more confident in the neighborhood of x after retraining. But
what if there are no items in the neighborhood of x? Actually this
example, despite its model uncertainty, doesn’t give us enough
information for faster model convergence, so we have to exclude such
outliers from our search space. This can be done by maximizing:

i(x) = -1/|X| ∑ⱼ₌₁ᴺ (||x — xⱼ||)

also called the informational density of an example. By maximizing
i(x), we pick only those points that are densely surrounded by other
points in our feature space.

Uncertainty measures coupled with informational density allow us to
pick points which are excellent representatives and still gives new
knowledge to the model. But if there are two representative points
where model is unconfident, how do you choose one? One way is to
choose the most dissimilar example to those already picked. The last
ingredient informative measure is diversity:

d(x) = 1 - max_{j \in train set}(\|x-x_j\|)

that promotes to pick as many diverse points as possible.

To combine these 3 measures in one scoring function, it is possible to
use the product of theirs corresponding cumulative distribution
functions (CDFs) (otherwise you should deal with numbers at the
different scales or even negative numbers):

Scoring Function: S(x) = U(x)*I(x)*D(x)

where each capital letter function denotes CDF, e.g. 
**U(x) = P(u < u(x))**

# Getting results faster

Finally, we’ve picked the example and can label it (there are many
data labeling tools available. The last step is to train the model. It
can be a long wait of hours until the model finishes it’s retraining
process and we can proceed with the next sample.

So what about if we do NOT train at all? Sounds crazy, but, if your
feature space exists without any additional parameters to be learned,
then you can rely on embeddings it produces and use simple metrics
like euclidean and cosine distances. This technique is well known as a
k-Nearest Neighbor classifier: for any sample, assign unknown class
from closest neighbor’s class (or from k closest neighbors).

Despite its simplicity, it is a very powerful method since: 

- It enables “learning” the new model instantly — actually no learning is
  performed, the labeled example is just stored in some efficient data
  structure (unless you are using brute-force search)

- It produces a non-linear model

- It can be easily scaled and deployed without model retraining — simply
  add new vectors to your running model

- If simple distance like euclidean is not enough, new metric spaces
  could be learned offline (see an overview of some metric learning
  methods as well as few-shot learning

- It also offers some explainability: when classifying some unknown
  example, it is always possible to investigate what has influenced this
  prediction (since we have access to all training set, that is
  infeasible only with learned parameters) 

Here is a visualisation how it looks in practice:

![./imgs/clusters.png](Clusters)

There are 2 classes, purples (p), and oranges (o). At each step, our
selection method tries to do the best to get the most informative
example and label it. As you can see, the model always tries to pick
the next sample in between 1st and 2nd class points, mainly because
maximizing uncertainty measure u(x) leads to minimizing

\|x - x_o\| - \|x - x_p\| \rightarrow \min

That leads to border exploration (around X=0) during ~30 first
steps. Then our active selection methods try to dive into a dense
cluster of points, thanks for informational density. Moreover, it’s
always aimed to explore different points due to diversity measure,
that is way better: conventional active learning gets stuck on only
one borderline and could neglect different point clusters.

# Validation and Results

To validate this active sampling method, let’s run an experiment on
the real dataset of fresh/rotten fruits images.

There are 3 types of fruits (apples, bananas, and oranges), each class
is exposed by fresh and rotten pictures, so there are 6 classes in
total. 90% of the dataset is retained for training purposes, and the
rest is for testing. To mitigate statistical errors, K-fold
cross-validation is applied with K=16, and test errors are averaged
afterward.

We take the 1280-dimensional output of MobileNet-v2 neural network as
embeddings for our feature space and adopt cosine distance for metric
space. Here are some insights on how the embedded dataset looks like

![./imgs/clusters.png](Clusters)

You can see that there are some cloud clusters so that we can benefit
here by using additional informativeness measures besides uncertainty.

![./imgs/random_vs_active_1.png](Random vs Active Selection)

The averaged model performance against the number of labeled examples
is depicted here. As we can see, active sampling strategy needs almost
2x times less labelling tasks to reach 90% accuracy compared with the
strategy, where points are randomly selected for labeling.

Another experiment was conducted on textual data, taken from
Sentiment140 dataset. The task consists of classifying tweets into
positive/negative sentiment. For experimentation purposes, we are
randomly sampling 20k points and do the same experiment as with
fresh/rotten fruit images above. As a feature extractor, Universal
Sentence Encoder is taken that produces 512-dimensional embeddings for
each tweet.

![./imgs/random_vs_active_2.png](Random vs Active Selection)

Similarly, here is how the model performs over labeling steps
comparing active & random selection strategies. Again, active
selection supposed to converge much faster, though the absolute
performance is lower (only 62% of accuracy and 70% is the best
accuracy when you use the full dataset).

# Summary

- When the labeling process is the bottleneck to build your next machine
  learning project, use active learning to minimize the number of
  labeling tasks

- Use pre-trained deep neural network’s outputs to convert your tasks
  from raw data (images, texts) to vectors (embeddings) 

- Apply a combination of informativeness measures to pick next training
  samples, to decrease model uncertainty, promote representativeness and
  diversity

- Choose k-NN as your classifier when you want to do instant training 
  and fast and transparent prediction 

- Compare your active learning results on each step with uniform
  sampling strategy on holdout dataset to see how performance evolves
  over picking steps and how you can economize your labeling budget
