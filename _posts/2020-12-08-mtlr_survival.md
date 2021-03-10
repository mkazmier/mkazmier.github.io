---
categories: ml
date: 2020-12-08
layout: post
title: Predicting survival using multi-task logistic regression
excerpt: Exploring an interesting ML algorithm for predicting survival-type outcomes.
math: true
---

In this post, I will describe an interesting machine learning algorithm for
survival prediction I\'ve been playing around with recently. This post will be
fairly technical --- if you just want to see it in action, you can head
stratight to the [Colab
notebook](https://colab.research.google.com/drive/1CkIEzctAE8_WPsXQdt2xPkY1kwbN3BSX?usp=sharing)
or check out my PyTorch implementation on
[Github](https://github.com/mkazmier/torchmtlr).

### Contents
-   [Survival analysis](#survival-analysis)
-   [Notation and mathematical
    preliminaries](#notation-and-mathematical-preliminaries)
-   [Learning survival distributions using multi-task logistic
    regression](#learning-survival-distributions-using-multi-task-logistic-regression)
    -   [The likelihood function](#the-likelihood-function)
    -   [Dealing with censoring](#dealing-with-censoring)
-   [Alternative derivation as a probabilistic graphical
    model](#alternative-derivation-as-a-probabilistic-graphical-model)
-   [Using MTLR](#using-mtlr)
-   [Conclusions](#conclusions)
-   [References & useful resources](#references-useful-resources)
-   [Footnotes](#footnotes)

## Survival analysis

In many real world problems, we\'re interested in predicting the time until a
certain event occurs. For example, we might be interested in the survival of a
cancer patient, the time until some part of a machine needs replacement, or the
time before a customer drops out. One unique aspect of the problem, which
warrants the use of specialized methods is *censoring* --- a particular form of
the missing data problem. Suppose we have a dataset of cancer patients and are
interested in predicting the time until cancer-related death. For patients who
already died at the time of analysis, the solution is straightforward: we could
just obtain their date of death (assuming a sufficiently complete and accurate
registry) and treat it as a supervised learning problem. However, what about
patients whose death we haven\'t observed (called *censored* in survival
analysis parlance), for example because they\'re still alive, or moved out to
another country? Excluding those cases could significantly reduce the size of
our training data and might introduce bias into the analysis. Furthermore,
censored observations still carry partial information --- we know for sure that
someone who is still alive after, say, a 5-year study have not died before that
time. Censoring is a central issue in survival analysis, and many of the methods
and algorithms are motivated by trying to handle it in a sensible way.

## Notation and mathematical preliminaries

We\'ll assume we have access to a dataset $$D$$ with $$N$$ samples. Each sample
is represented by a 3-tuple $$(T^{(n)}, \delta^{(n)}, \mathbf{x}^{(n)})$$,
where:

-   $$T^{(n)}$$ is the time to event (e.g. death) or censoring,
-   $$\delta^{(n)}$$ is the event indicator---it\'s equal to 1 if the
    event was observed and 0 if the instance was censored,
-   $$\mathbf{x}^{(n)}$$ is a vector of features (e.g. age, cancer stage,
    gene expression levels, etc.).

You\'ll often see me refer to data instances as patients, since I\'ve been
working primarily with medical data recently. However, the same principles apply
to other types of instances and events, e.g. machines failing or customers
dropping out.

The *probability density function* (PDF) $$f(t)$$ defines the probability of
event occuring in some particular interval:

$$
P(t_i \le T \le t_j) = \int_{t_i}^{t_j} f(\tau) d\tau,\ t_i < t_j.
$$

If we treat time as discrete (e.g. advancing in one month intervals), this
becomes the probability mass function $$P(T = t)$$ instead.

The *survival function* is the probability of surviving longer than some time
$$t$$:

$$
S(t) = P(T > t) = 1 - F(t),\ 0 \le t < \infty, 
$$

where $$F(t) = \int_0^t f(\tau) d\tau$$ is the cumulative distribution function
(CDF). Or, in the discrete case:

$$
S(t) = 1 - \sum_{k=0}^t P(T = t_k),\ 0 \le t < \infty, 
$$

Note that $$S(0) = 1$$. We\'ll be primarily interested in the survival function
here, since it lets us answer questions like \'What is the probability that this
patient will survive longer than 2 years?\'

## Learning survival distributions using multi-task logistic regression

So how could we learn to predict survival given some features $$\mathbf{x}$$?
Let's assume there are no censored observations for now --- we will deal with
the general case shortly. One idea would be to fix a timepoint, say $$t^*$$ and
treat the problem as binary classification. We define $$y = 0$$ if $$T < t^*$$
and $$y = 1$$ if $$T \ge t^*$$ and use [logistic
regression](https://en.wikipedia.org/wiki/Logistic_regression) to predict
whether a patient survives longer than $$t^*$$:

$$
 P_{\boldsymbol{\theta}}(y = 1 \mid \mathbf{x}) = \frac{1}{1+\exp(-(\boldsymbol{\theta}^T\mathbf{x} +
b))},
$$

where $$\boldsymbol{\theta}, b$$ are trainable parameters. One advantage of this
approach is simplicity --- it works pretty much out of the box with any logistic
regression implementation, no need to roll out any custom algorithms. A big
issue though is that binarization can obscure important information. As an
example, consider 2 cancer patients: one who lives for 2 years and 1 month and
another surviving for 15 years after diagnosis. It seems obvious that these are
quite different cases, but from the perspective of the binary model they both
fall in the same class. This makes learning more difficult, as dissimilar
samples are forced together into one class, and the predictions themselves are
less informative. Also, for many problems it might not be obvious what cutoff
time to choose. Ideally, we would like not to choose at all and make use of all
the available information.

A simple extension of the above approach is to consider several different
timepoints (say, $$K+1$$), dividing the time axis into discrete intervals and
then using logistic regression to predict the probability that the event occurs
within each interval.

{:refdef: style="text-align: center;"}
![](/assets/images/mtlr_time_discretization.svg)
{: refdef}

That is, in each time interval $$[t_{k-1}, t_k)$$ we solve a binary
classification problem:

$$
P_{\boldsymbol{\theta}_i}(y_k = 1 \mid \mathbf{x}) =
\frac{1}{1+\exp(-(\boldsymbol{\theta}_k^T\mathbf{x} + b_k))},\ 1 \le k \le K,
$$

where $$y_k = 1$$ if $$T < t_k$$, i.e. if the patient experienced an
event prior to $$t_k$$. We use a separate set of parameters
$$(\boldsymbol{\theta}_k, b_k)$$ at each timepoint, to capture the potentially
time-varying effect of features. This can be represented as a binary
sequence of length $$K$$, with 0s and 1s used as above.

{:refdef: style="text-align: center;"}
![](/assets/images/mtlr_encoding.svg)
{: refdef}

The sequence notation is simply more convenient to use in derivations --- you
should remember that probabilities of these binary sequences map directly to
probabilites of events:

$$
P((y_1 = 0, \dots, y_{k-1}=0, y_k = 1, \dots, y_K = 1)) = P(t_{k-1} \le T < t_k).
$$

We have turned the task of survival prediction into a series of binary
classification tasks, which we solve using logistic regression. To train the
model, we now need to derive the likelihood function.

### The likelihood function

If the probability of event at each timepoint was independent from all the other
timepoints, the join probability of a sequence $$\mathbf{y}$$ would simply be
the product of the individual probabilites at each timepoint:

$$
P_{\boldsymbol{\Theta}}(\mathbf{y} = (y_1, y_2, \dots, y_{K-1}) \mid \mathbf{x}) =
P_{\boldsymbol{\theta}_1}(y_1 \mid \mathbf{x})P_{\boldsymbol{\theta}_2}(y_2
\mid \mathbf{x})\dots P_{\boldsymbol{\theta}_{K-1}}(y_{K-1}\mid \mathbf{x}).
$$

To evaluate this product, we will use two properties of the logistic sigmoid
function:

1.  $$\frac{1}{1+\exp(-u)} = \frac{\exp(u)}{1+\exp{u}}$$ and
2.  $$1 - \frac{1}{1+\exp(-u)} = \frac{1}{1+\exp(u)}$$.

First, write the predicted probability of $$y = 0$$ and $$y = 1$$ as

$$ 
\begin{align}
P_{\boldsymbol{\theta}_k}(y_k = 1 \mid \mathbf{x}) &=
\frac{1}{1+\exp(-(\boldsymbol{\theta}_k^T\mathbf{x} + b_k))}\\ &=
\frac{\exp((\boldsymbol{\theta}_k^T\mathbf{x} +
b_k)y_k)}{1+\exp(\boldsymbol{\theta}_k^T\mathbf{x} + b_k)} &  \text{(Using property 1 and }y_k=1\text{)}
\end{align}
$$

and

$$
\begin{align}
P_{\boldsymbol{\theta}_k}(y_k = 0 \mid \mathbf{x}) &= 1-P_{\boldsymbol{\theta}_k}(y_k = 1 \mid \mathbf{x})\\
 &= 1 - \frac{1}{1+\exp(-(\boldsymbol{\theta}_k^T\mathbf{x} + b_k))}\\ 
&= \frac{1}{1+\exp(\boldsymbol{\theta}_k^T\mathbf{x} + b_k)} &  \text{(Using property 2)}\\
&= \frac{\exp((\boldsymbol{\theta}_k^T\mathbf{x} + b_k)y_k)}{1+\exp(\boldsymbol{\theta}_k^T\mathbf{x} + b_k)} & \text{(Using property 1 and }y_k=0\text{)}.
\end{align} 
$$

This lets us write the joint density as

$$
\begin{align}
P_{\boldsymbol{\Theta}}(\mathbf{y} = (y_1, y_2, \dots, y_{K-1}) \mid \mathbf{x}) &=
\frac{\exp((\boldsymbol{\theta}_1^T\mathbf{x} + b_1)y_1)}{1+\exp(\boldsymbol{\theta}_1^T\mathbf{x} + b_1)} \times
 \dots \times
\frac{\exp((\boldsymbol{\theta}_{K-1}^T\mathbf{x} + b_{K-1})y_{K-1})}{1+\exp(\boldsymbol{\theta}_{K-1}^T\mathbf{x} + b_{K-1})}\\
 &= \frac{\exp(\sum_{k=1}^{K-1}((\boldsymbol{\theta}_k^T \mathbf{x} + b_k)y_k))}
{\prod_{k=1}^{K-1}(1 + \exp(\mathbf{\theta}_k^T \mathbf{x} + b_k))}.
\end{align}
$$

There is a problem with this derivation, however. Going back to the cancer
example, if a patient dies at time $$t_i$$, they cannot come back to life at any
time after that [^1]. This means that certain sequences of $$y$$s **will never
appear** (specifically, any sequences where for some $$i,\ y_i = 1$$ and $$y_{i+1} =
0$$), meaning the formula above underestimates the true event probability. In
fact, there are only $$K$$ legal sequences, including all 0s and all 1s (you can
try enumerating all valid encodings for some small K as an exercise). We can fix
this by changing the denominator to only normalize by the scores of allowed
sequences:

$$
P_{\boldsymbol{\Theta}}(\mathbf{y} = (y_1, y_2, \dots, y_{K-1}) \mid \mathbf{x}) =
\frac{\exp(\sum_{k=1}^{K-1}((\boldsymbol{\theta}_k^T \mathbf{x} +
b_k)y_k))}{\sum_{i=1}^{K}\exp(\sum_{k=i}^{K-1}\boldsymbol{\theta}_k^T
\mathbf{x} + b_k)},
$$

which makes the individual regressors implicitly dependent.

The (uncensored) log-likelihood is now straightforward to derive: given a
dataset $$D = \{T^{(j)}, \delta^{(j)}, \mathbf{x}^{(j)}\}_{j=1}^{N}$$ we have

$$
\begin{align*}
L(\boldsymbol{\Theta}, D) &= \log(\prod_{j=0}^N P_{\boldsymbol{\Theta}}(\mathbf{y}^{(j)}\mid \mathbf{x}^{(j)}))\\
&= \sum_{j=1}^{N}\sum_{k=1}^{K-1}((\boldsymbol{\theta}_k^T \mathbf{x}^{(j)} +
b_k)y_k^{(j)} - \log(\sum_{i=1}^{K}\exp(\sum_{k=i}^{K-1}\boldsymbol{\theta}_k^T
\mathbf{x}^{(j)} + b_k))), 
\end{align*}
$$

with the $$y$$s defined as above.

### Dealing with censoring

Recall that in survival analysis we usually have to deal with censoring, i.e. we
don\'t know the true time of event for some patients. The key insight, used in
many survival analysis algorithms, is that censored patients still provide
partial information --- namely that they did not experience the event before the
last time we knew their status. The original MTLR publication uses this to
propose an elegant method of dealing with censoring. Let\'s have a look at an
example survival encoding $$\mathbf{y}$$ for a censored instance:

{:refdef: style="text-align: center;"}
![](/assets/images/mtlr_censoring.svg)
{: refdef}

The ?s denote unknown status --- they could be either 1 or 0, since we
don\'t know the true time of death, only that it was no earlier than
$$T_c$$. Based on this data alone, all of these sequences are possible:

$$
\begin{align*}
&(0,0,0,0,0,0),\\
&(0,0,0,0,0,1),\\
&(0,0,0,0,1,1),\\
&(0,0,0,1,1,1),\\
&(0,0,1,1,1,1)
\end{align*}
$$

Under the assumption of independent censoring (censored patients are no more or
less likely to experience an event than others given their covariates), all of
these sequences are equally likely. We can compute the likelihood of this
instance by simply summing (or *marginalizing*) over all sequences
$$\mathbf{y}$$ compatible with the observation:

$$
\begin{align}
P_{\boldsymbol{\Theta}}(\mathbf{y} = (y_1, y_2, \dots, y_{K-1}) \mid \mathbf{x})
&=  P_{\boldsymbol{\Theta}}(\mathbf{y} = (y_1=0, \dots, y_j=0, y_{j+1}=0, \dots, y_{K-1}=0) \mid \mathbf{x})\\
&+ P_{\boldsymbol{\Theta}}(\mathbf{y} = (y_1=0, \dots, y_j=0, y_{j+1}=0, \dots, y_{K-1}=1) \mid \mathbf{x})\\
&+ \dots + \\
&+ P_{\boldsymbol{\Theta}}(\mathbf{y} = (y_1=0, \dots, y_j=1, y_{j+1}=1, \dots, y_{K-1}=1) \mid \mathbf{x})\\
&= \sum_{i=j}^{K-1}\frac{\exp(\sum_{k=i}^{K-1}((\boldsymbol{\theta}_k^T \mathbf{x} + b_k)y_k))}{\sum_{k=1}^{K}(1 + \exp(\mathbf{\theta}_k^T \mathbf{x} + b_k))}
\end{align}
$$

where $$t_j = T_c$$. For the example above, this works out to

$$
\begin{align*}
 &P_{\boldsymbol{\Theta}}(\mathbf{y} = (0,0,0,0,0,0) \mid \mathbf{x})\\
 +&P_{\boldsymbol{\Theta}}(\mathbf{y} = (0,0,0,0,0,1) \mid \mathbf{x})\\
 +&P_{\boldsymbol{\Theta}}(\mathbf{y} = (0,0,0,0,1,1) \mid \mathbf{x})\\
 +&P_{\boldsymbol{\Theta}}(\mathbf{y} = (0,0,0,1,1,1) \mid \mathbf{x})\\
 +&P_{\boldsymbol{\Theta}}(\mathbf{y} = (0,0,1,1,1,1) \mid \mathbf{x})
\end{align*}
$$

Combining that with the uncensored log-likelihood from above, the full
log-likelihood for a dataset $$D$$ with $$N$$ instances, $$N_c$$ of which are
censored, is

$$
\begin{align*}
L(\boldsymbol{\Theta}, D) &=
\sum_{j=1}^{N-N_c}\sum_{k=1}^{K-1}(\boldsymbol{\theta}_k^T \mathbf{x}^{(j)} +
b_k)y_k^{(j)} &\text{(Uncensored)}\\
 &+ \sum_{j=N-N_c+1}^{N}\log(\sum_{i=1}^{K-1}\mathbf{1}\{t_i \ge T_c^{(j)}\}\exp(\sum_{k=i}^{K-1}((\boldsymbol{\theta}_k^T\mathbf{x}^{(j)} + b_k)y_k^{(j)}))) &\text{(Censored)}\\
 &- \sum_{j=1}^{N}\log(\sum_{i=1}^{K}
\exp(\sum_{k=i}^{K-1}\boldsymbol{\theta}_k^T \mathbf{x}^{(j)} + b_k)), &\text{(Normalizing constant)}
\end{align*}
$$

where $$\mathbf{1}\{\text{cond}\} = 1$$ if the condition in brackets is
satisfied and 0 otherwise, and $$T_c$$ is the time of censoring.

To mitigate overfitting, we usually add $$\ell_2$$ regularization:

$$
L(\boldsymbol{\Theta}, D)_{\mathrm{reg}} = L(\boldsymbol{\Theta}, D) + \frac{C_1}{2}\sum_{k=1}^{K-1}\Vert\boldsymbol{\theta}_k\Vert_2^2.
$$

Here, $$C_1$$ is a hyperparameter determining the strength of regularization. It
also controls how much the parameters change between consecutive timepoints and
hence the smoothness of the predicted survival curves (see the Appendix
[here](https://era.library.ualberta.ca/items/3deb4dd9-788d-4c61-94a5-d3ee6645f74f)
for proof).

## Alternative derivation as a probabilistic graphical model

*(Note: this section is not required for basic understanding.)*

MTLR can also be viewed from the perspective of probabilistic graphical models.
A *conditional random field* (CRF) is an undirected graphical model which is
essentially an extension of logistic regression to problems with structured
outputs. In particular, MTLR can be viewed as an instance of *linear chain CRF*,
often used for sequence labeling tasks, corresponding to the following graphical
model:

{:refdef: style="text-align: center;"}
![](/assets/images/mtlr_crf.svg)
{: refdef}

The joint probability density of a generic linear chain CRF is:

$$
P_{\boldsymbol{\theta}}(\mathbf{y} = (y_1, y_2, \dots, y_{T}) \mid \mathbf{x}) =
\frac{1}{Z(\mathbf{x}, \boldsymbol{\theta})}\prod_{t=1}^T \phi(y_k \mid \mathbf{x},
\boldsymbol{\theta})\prod_{t=1}^{T-1} \phi(y_k, y_{k+1} \mid \mathbf{x}, \boldsymbol{\theta}),
$$

where the $$\phi$$s are the node and edge *potential functions*[^2] and $$Z$$ is
the normalizing constant (also known as *partition function*) which ensures that
the sequence-level probabilities sum to 1:

$$
Z(\mathbf{x},
\boldsymbol{\theta}) = \sum_{y'} \prod_{t=1}^T \phi(y'_k \mid \mathbf{x},
\boldsymbol{\theta})\prod_{t=1}^{T-1} \phi(y'_k, y'_{k+1} \mid \mathbf{x},
\boldsymbol{\theta}).
$$

(See [this blog
post](https://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)
and [Kevin Murphy\'s textbook](https://www.cs.ubc.ca/~murphyk/MLbook/)
for more information about CRFs.)

To obtain the MTLR model from the general CRF, we\'ll use the following
potential functions:

$$
\phi(y_k \mid \mathbf{x}, \boldsymbol{\theta}_k) =
\exp(y_k(\boldsymbol{\theta}_k^T\mathbf{x} + b_k))
$$

and

$$
\phi(y_k, y_{k+1} \mid \mathbf{x}, \boldsymbol{\theta}_k) =
\mathbf{1}\{y_k = 0 \lor y_{k+1} = 1\},
$$

(the role of the second potential function will become clear soon). Note that
while most CRFs used in e.g. natural language processing tend to share
parameters over the entire sequence, here we use a separate weight vector for
each node to capture the time-varying effect of features.

Plugging the above potentials into the above density gives

$$
\begin{align}
P_{\boldsymbol{\theta}}(\mathbf{y} = (y_1, y_2, \dots, y_{T}) \mid \mathbf{x}) &=
\frac{1}{Z(\mathbf{x}, \boldsymbol{\theta})}\prod_{k=1}^{K-1}
\exp(y_k(\boldsymbol{\theta}_k^T\mathbf{x} + b_k))\prod_{k=1}^{K-2} \mathbf{1}\{y_k =
0 \lor y_{k+1} = 1\} \\
&= \frac{1}{Z(\mathbf{x}, \boldsymbol{\theta})}
\exp(\sum_{k=1}^{K-1} y_k(\boldsymbol{\theta}_k^T\mathbf{x} + b_k))\prod_{k=1}^{K-2}\mathbf{1}\{y_k =
0 \lor y_{k+1} = 1\}.
\end{align}
$$

Looking at the edge potential term, we can see that the product will be equal to
1 only if the condition holds for all pairs of consecutive $$y_k$$s, i.e. if
$$(y_k = 1,\ y_{k+1} = 0)$$ doesn\'t occur anywhere in the sequence. If you think
back to how we defined a valid sequence, this choice of edge potential ensures
that cases where a patient \"comes back to life\" are assigned 0 probability,
that is

$$
P_{\boldsymbol{\theta}}(\mathbf{y} = (y_1, y_2, \dots, y_{T}) \mid \mathbf{x}) =
\begin{cases}
\frac{1}{Z(\mathbf{x}, \boldsymbol{\theta})} 
\exp(\sum_{k=1}^{K-1} y_k(\boldsymbol{\theta}_k^T\mathbf{x} + b_k))& \text{if } \mathbf{y} \text{ is a valid sequence}\\
0 & \text{otherwise.}
\end{cases}
$$

This also lets us write the normalizing constant as

$$
\begin{align}
Z(\mathbf{x}, \boldsymbol{\theta}) &= \sum_{y'}
\exp(\sum_{k=1}^{K-1} y'_k(\boldsymbol{\theta}_k^T\mathbf{x} + b_k))\prod_{k=1}^{K-2}
\mathbf{1}\{y'_k = 0 \lor y'_{k+1} = 1\}\\
&= \sum_{y' \text{valid}}
\exp(\sum_{k=1}^{K-1} y'_k(\boldsymbol{\theta}_k^T\mathbf{x} + b_k)),
\end{align}
$$

which you should recognize as the same form as the denominator in the MTLR density
from the previous section.

While the original formulation enforced it implicitly by assuming that only
valid sequences occur in real-world data and deriving the normalization constant
under this assumption, the graphical model formulation enforces the validity of
$$\mathbf{y}$$ explicitly through the use of edge potentials. Additionally, it
lets us interpret the MTLR approach to dealing with censored data as
marginalization over the unobserved $$y_k$$ nodes in the graph (see Chapter 19
in [Koller &
Friedman](https://mitpress.mit.edu/books/probabilistic-graphical-models), where
our independent censoring assumption corresponds to their missing-at-random
assumption).

## Using MTLR

The model described above can be used to learn flexible, potentially
time-varying survival functions as it is. However, since it\'s differentiable
end-to-end, there are many more exciting possibilites. You can effectively use
it as a survival prediction \'head\' in any neural net, just like you would use
a logistic regression layer for binary classification --- for example on top of
a CNN to learn to predict survival directly from medical images (something I\'m
working on now), an LSTM to handle time-varying features or something more
exotic like a [neural ODE](http://arxiv.org/abs/1907.03907).

## Conclusions

Hopefully you now have a better understanding of the theory behind MTLR and are
ready to solve some survival analysis problems! Make sure to check out the
[Colab
notebook](https://colab.research.google.com/drive/1CkIEzctAE8_WPsXQdt2xPkY1kwbN3BSX?usp=sharing)
accompanying this post for examples using both linear and deep MTLR. I\'m also
working on [a standalone PyTorch-based
package](https://github.com/mkazmier/torchmtlr), which aims to be modular,
extensible and easy to use in any PyTorch model.

## References & useful resources

-   [The original NeurIPS
    publication](https://papers.nips.cc/paper/4210-learning-patient-specific-cancer-survival-distributions-as-a-sequence-of-dependent-regressors)
    for theoretical background and [Ping Jin\'s master\'s
    thesis](https://era.library.ualberta.ca/items/3deb4dd9-788d-4c61-94a5-d3ee6645f74f)
    for various theoretical insights and implementation tricks,
    including reformulation as softmax classifier and a proof that the
    second regularizer from the original paper is redundant.
-   [This paper](https://arxiv.org/abs/1801.05512) which is the first
    application of Deep MTLR (as far as I know).
-   [This paper](https://arxiv.org/abs/2003.08573v2) on Bayesian variant
    of Deep MTLR that allows for better handling of uncertainty and
    out-of-distribution samples.
-   [PySurvival](https://square.github.io/pysurvival/) has PyTorch
    implementations of both MTLR and Deep MTLR as well as other useful
    survival models.

## Footnotes

[^1]: Some types of events can occur more than once per lifetime, for
    example cancer recurrence or repeated illness. However, here we\'re
    restricting the analysis to non-recurrent, terminal events (like
    death).

[^2]: Note that the potential functions need not actually correspond to
    probability distributions, i.e. can take arbitrary values. The global
    normalizing constant ensures that the model defines a valid probability
    distribution.
