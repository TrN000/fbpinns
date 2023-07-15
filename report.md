---
title: Report on Finite Basis PINNs
author: Nicolas Trutmann
date: \today
legi: "14-913-552"
abstract: |

  In this report I replicate parts of the findings of Moseley, Markham and Nissen-Meyer
  from their paper  [Finite Basis Physics-Informed Neural Networksj (FBPINNs) a scalable
  domain decomposition approach for solving differential
  equations](https://arxiv.org/abs/2107.07871)

...

## Contributions

I wrote this project alone.

## Intoduction

[FBPINNs](https://arxiv.org/abs/2107.07871)

The code to this paper is available on [GitHub](https://github.com/TrN000/fbpinns).
Instructions for installing and running the code are provided in the readme.

## Motivation

large-scale problems

high frequency problems

computationally expensive to use FCN with large hidden layers.

## Methods

Simple FCN model used throughout. "large" model for naïve approach: 5 hidden 128 neurons
"small" for FBPINN: 2 hidden 16 neurons.

same window, activation function.

Adam with learning rate of 1e-3 and 10000 learning steps(due to machine and time
limitations)

total parameters: 9630 vs 66433

### Implementation Details

An implementation detail of note is the following: I developed a simple but effective way
to combine models into larger meta-models.  Suppose you had identically dimensioned
models, you could want to sum them together or multiply them in a generic way. I wrote two
model types based on the `torch.nn.ModuleList` module wich have basic sums and products as
their forward method.  Here is the `fbpinns.net.Additive` model:

```python
class Additive(nn.ModuleList):
    def __init__(self, *args):
        super(Additive, self).__init__(*args)

    def forward(self, x: Tensor) -> Tensor:
        acc = torch.zeros_like(x)
        for module in self:
            acc += module(x)
        return acc
```

The `fbpinns.net.Multiplicative` model is designed in a similar fashion.  It provides a
clear way to express the relationship between models.  I used the multiplicative version
to combine a FCN with a window function and those in turn I combined to sum to equation
(13) from the paper.  I found nothing like it in a cursory search on the topic, so I
include it here.


## Results

In the initial stages of the implementation I found that, while the FBPINN approach does
approximate the PDE with high accuracy(low loss), the difference in absolute terms can be
very large far away from the boundary(see figure TODO: initial approach). Which confirms
the findings of the paper.
