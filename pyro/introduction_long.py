"""
Note this tutorial skips some plots, the code focus is on pyro. See link for full tutorial
https://pyro.ai/examples/intro_long.html

Excellent supplementary reading: https://medium.com/paper-club/understanding-pyros-model-and-guide-a-love-story-7f3aa0603886

"probabilistic program": generative process for data based on deterministic computation and randomly sampled values
"inference": inference is simply deriving the probability of a random variable taking a value or set of values, P(X = x).
             In the ML landscape, it is typically the task to compute the posterior distribution of a variables probability
             given some evidence, e.g. P(θ|x). We infer from the data on the data-generating process.

Pyro is build on PyTorch.

BACKGROUND:
    We have some model with parameters theta, p_t(x, z) = p_t(x|z)p_t(z).
    x is observed data and z are hidden latent states
    i.e. the joint probability of x and z (given parameteres theta) is equal to the
    conditional probability of x given z multiplied by z (see law of total probability)

    p_t(z) is our prior (i.e. probability of observing z)
    see the article for graphical notation

    We know from probability theory that P(A|B) = P(A ∩ B) / P(B)
    therefore
    P(z | x) = P(x, z) / P(z), P(z) = ∫ P(x, z) dz
    where P is the probability distribution given some parameters theta
    and ∫ P(x, z) dz calculates the probability of x by margainalising over all z

    CHECKING RESULTS SECTION

"""
import logging
import os

import torch             # (pytorch, META)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

# cont_africa is bool, rugged is ruggeddness terrain scores,
# rgdppc_2000 is real GDP per capita for 2000.
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
dataframe = data[["cont_africa", "rugged", "rgdppc_2000"]]
dataframe["rgdppc_2000"] = np.log(dataframe["rgdppc_2000"])

# convert to torch.tensor
train = torch.tensor(dataframe.values, dtype=torch.float)
is_cont_africa, ruggedness, gdp = train[:, 0], train[:, 1], train[:, 2]



"""
Models in Pyro

Probibalistic models in python (e.g. probability distriubtions) are specified
as python functions. These generate data from latent variables using special 
primative functions (e.g. param, sample, plate).

"""
# In our example, we can

def simple_model(is_cont_africa, ruggedness, gdp=None):
    a = pyro.param("a", lambda: torch.randn(()))              # intercept
    b_a = pyro.param("bA", lambda: torch.randn(()))           # Beta is_cont
    b_r = pyro.param("bR", lambda: torch.randn(()))           # Beta ruggedness
    b_ar = pyro.param("bAR", lambda: torch.randn(()))          # Beta interaction
    sigma = pyro.param("sigma", lambda: torch.ones(()),       # torch.ones(()) = 1, variance parameter
                       constraint=dist.constraints.positive)

    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

    # mean is a obs.size x 1 array, sigma = 1
    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=gdp)

# this is awesome visualisation, requires sudo apt-get install graphviz, saves to PDF
graph = pyro.render_model(simple_model, model_args=(is_cont_africa, ruggedness, gdp),
                          render_distributions=True, render_params=True)
graph.render()


# pyro.sample ----------------------------------------------------------------------------------------------------------

# Pyro sample allows us to randomly sample from a given distribution. e.g.
# To start we will sample P(θ) as a gaussian probability distribution

normal_one = dist.Normal(loc=0, scale=1).sample()  # we can sample from a distribution like this

normal = pyro.sample(name="obs",                                    # str, primitive must have a name
                     fn=dist.Normal(loc=0, scale=1),  # pass a distribution with parameters
                     sample_shape=(10, 1))                          # shape of samples to draw

# normal is a torch array size [10, 1]

# IMPORTANT 1
# Note the critical "obs" keyword. This has a deep behaviour. When simple model
# is given a gdp argument as obs, sample will always return gdp
# HOWEVER, "when any sample statement is observed, the cumulative effect of every
# other sample statement in a model changes, following Bayes rule". Mathematically
# consistent values are assigned to all pyro.sample statements in a model

# IMPORTANT 2
# We can also condition directly on a model with pyro.condition (see docs) rather than provide obs

# pyro.param -----------------------------------------------------------------------------------------------------------

# This is a convenient way to store paramteres. They will persist across model calls unless specifically updated
# They can be subject to constraints (see sigma, variance must be positive) - very cool!


# pyro.plate -----------------------------------------------------------------------------------------------------------

# see "plate notation" : https://en.wikipedia.org/wiki/Plate_notation

# similar to vectorised for loop.


# Example: from maximum-likelihood regression to Bayesian regression ----------------------------------------------------

# Currently we have a likelihood, p(x | θ). To go from MLE to MAP we need to give a prior over θ,
# p(z | x) = p(x | θ)p(θ) / p(x)
# To do this, rather than taking the parameters as fixed values (drawn randomly) as above,
# the parameters are drawn from prior distributions, which are parameterised by hyperparameters

def model(is_cont_africa, ruggedness, gdp=None):
    a = pyro.sample("a", dist.Normal(0, 10.0))
    b_a = pyro.sample("bA", dist.Normal(0, 1))
    b_r = pyro.sample("bR", dist.Normal(0, 1))
    b_ar = pyro.sample("bAR", distr.Normal(0, 1))
    sigma = pyro.sample("sigma", dist.uniform(0, 10.0))  # still constrained > 0
    
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    
    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=gdp)
    
"""
Inference : Variational Bayes

Unified scheme for:
    1) finding θmax (i.e. argmax θ pθ(x)) 
       i.e. the theta set that maximise the likelihood of observing
       our set of data x
       
    2) computing a tractable posterior  qθ(z) to the true pθ(z | x)

Essentially, we find a surrogate distribution qφ(z) 
(called the 'variational distribution' in the literature,
in pyro THE GUIDE) that minimize the KL divergence between qφ(z)
and our true posterior pθ(z | x). KL is minimized by ELBO (see notes).

IMPORTANT: the approximating distribution we use in pyro
is called the 'guide'. Guide and posterior model are 
linked by the "name" parameter.

We can add a variational distribution now. We will approximate
the posterior distribution with a Gaussian distribution with
diagonal covariance matrix (ie. all covar = 0). This is the 
"mean field approximation" (see notes).

Let's just recap exactly what we are doing. We have a 
set of data where each observation is or is not 
in africa, has a level of ruggedness and a GDP.

We want to model this, and we do so with the equation:
y = a + b1(is_africa) + b1(ruggedness) + b2(is_africa)(ruggedness)

We are interested in predicting y = log_gdp. We want p(θ|x) = p(x|θ)p(θ) / p(x)
and we convert to MAP, putting priors over all θ. This is out posterior
p(θ | x). However, it is intractable. Therefore, we use variational bayes,
creating a guide distribution. For this guide distribution we will asssume it is 
Gaussian, and use it to approximate p(θ|x) = p(x|θ)p(θ) / p(x)
 """

def custom_guide(is_cont_africa, ruggedness, gdp=None):

    # Hyperparameters that control the priors on θ (i.e. p(θ))
    a_loc = pyro.param("a_loc", lambda: torch.tensor(0.0))
    a_scale = pyro.param("a_scale", lambda: torch.tensor(1.0))
    sigma_loc = pyro.param("sigma_loc", lambda: torch.tesnsor(1.0))
    weights_loc = pyro.param("weights_loc", lambda: torch.randn(3))
    weights_scale = pyro.param("weights_scale", lambda: torch.ones(3),
                               constraint=dist.constraints.positive)

    # Sampling from θ
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))

    # return the list of sampled parameters
    return {"a": a, "b_a": b_a, "b_r": b_r, "b_ar": b_ar, "sigma": sigma}

# Note these random variables are all separate from eachother, see
# docs guide for rendered image

# Note, we can use pyro's autoguide to make these variational distributions
# for many base distributions

 autogenerated_guide = pyro.infer.autoguide.AutoNormal(simple_model)
# autogenerated_guide is a function AutoNormal() that takes the same args as simple_model! cool

# Maximising ELBO ------------------------------------------------------------------------------------------------------

# EBLO = E_{qφ(z)} [log p0(x, z) - qφ(z)]
# i.e. the expected difference given some parameters φ
# evaluated at the latent variable z and our true posterior.

# "Optimizing the ELBO over model and guide parameters via stochastic
# gradient descent using these gradient estimates is sometimes called
# stochastic variational inference (SVI);"

# NOTES:
# https://mbernste.github.io/posts/variational_inference/
# https://fabiandablander.com/r/Variational-Inference.html
# https://sccn.ucsd.edu/~rapela/docs/vblr.pdf
# https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf
# https://rpubs.com/cakapourani/variational-bayes-lr
























