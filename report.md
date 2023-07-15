---
title: Report on Finite Basis PINNs
author: Nicolas Trutmann
date: \today
legi: "14-913-552"
abstract: |

  In this report I replicate parts of the findings of Moseley, Markham and Nissen-Meyer
  from their paper "Finite Basis Physics-Informed Neural Networksj (FBPINNs) a scalable
  domain decomposition approach for solving differential
  equations".

...

## Contributions and Declaration of Originality

I wrote this project alone. While I have found the code to the original paper online,
I wrote the entirety of the code myself and relied on the original code only for clarity
where the paper lacked it.

## Intoduction

The original paper is available here:
[FBPINNs](https://arxiv.org/abs/2107.07871)

The code to this paper is available on [GitHub](https://github.com/TrN000/fbpinns).
Instructions for installing and running the code are provided in the readme.

## Motivation

In recent years, physics-informed neural networks (PINNs) have lead to powerful methods
for solving differential equations. Despite their many advantages, they struggle in
several aspects. They struggle to approximate large-scale problems, where a single network
has to solve a PDE over a large domain. On multi-scale problems they also fail to capture
the entire solution, either approximating it only spotwise or poorly overall.

Finite basis PINNs promise to solve these problems. The name is borrowed from finite
element method from classical numerical methods, where the solution of the differential
equation is composed of a finite set of basis functions. The core idea is to partition the
problem domain into smaller patches, each of which is associated with its own neural
network, which is only responsible for approximating the PDE on its restricted domain.
This shows good results on both larger scales and high frequency problems, as the
responsibility for the solution is distributed over many smaller networks, which can
moreover be parallelized.
Additionally, since this distributes the complexity of the problem over many networks,
each individual network can be smaller, with fewer parameters, which lightens the
computational load.

## Methods

The paper bases all its networks on a simple fully connected neural network (FCN).  In the
section of the paper I replicate, the traditional approach is formed by a single, "large"
FCN with 5 hidden layers with 128 neurons each.  The finite basis neural network (FBN) is
composed of 30 separate FCN with 2 hidden layers each and 16 neurons.  This brings the
total free parameter count of the networks to 66433 and 9630 respectively.

I use the same activation function, `Tanh`, as the paper as well as the same optimizer,
Adam with learning rate of 1e-3, but only 10000 learning steps due to machine and time
limitations.

As a window function I attempted to implement the window function given in the paper
(equation 14), however, key parameters were ommited. I guessed them as best as possible.

As my own extension I first attempted to combine the results into a larger model with the
help of the `Additive` module that I wrote(see below). which combines the fully connected
model with the partitioned one and trains both at the same time.

Further investigating this result I developed a logarithmic approach by parttioning the
domain in logarithmic increments and adding these up.
In my experiment I added 4 layers, fully connected, partitioned into 2, 4 and 8
subdomains, all with 2 hidden layers and 16 neurons.

In total these networks have the following free parameters:

| model | free parameters |
| ----- | --------------- |
| FCN   | 66433 |
| FBN   | 9630 |
| Cumulative | 9951 |
| Logarithmic | 4815 |

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
very large far away from the boundary. Which confirms the findings of the paper.

![Iterations: 10'000. The loss shows clearly that the FBN has
approximated the PDE more accurately. Also visually, the actual solution shadows the
results of the FBN almost completely, while the FCN has noticeable defects.](./img/fb_final_10000.png)

My results differ from the paper in the respect that the window function plays an
important role in the performance of the FBN. This turns out to be very sensitive to the
amount of overlap and the overlap gradient of the windows.

![Early stages of a FBN during training. The local behavior of the PDE is observable after
only a few iterations, but the low-frequency part of the solution is not captured.
Iterations: 100](./img/fbpinn_100_2_16.png)

The extension models performed very well. Unsurprisingly the combined model managed to
beat my implementation of the FBN. What is astonishing is that it only required a FCN of 2
hidden layers and 16 neurons to correct for the wandering far out from the boundary.
The logarithmic approach on the other hand did not quite so well, however it had only half
the free parameters so a poorer performance was to be expected.

![Iterations: 1000. The FBN shows the usual wandering far out from the boundary, but this
behavior is absent in the cumulative solution. The logarithmic approach did not converge
at all, as the loss graph substantiates.](./img/fb_ext1000.png)
