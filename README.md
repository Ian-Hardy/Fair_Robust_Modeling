# Fair Robust Modeling

An implementation of the 2021 paper "To be Robust or to be Fair: Towards Fairness in Adversarial Training" (https://arxiv.org/pdf/2010.06121.pdf)

Super happy to share that this work won an Outstanding Paper Award in the ML Reproducibility Challenge 2021: https://openreview.net/forum?id=Sczshz7h0K

These resources were designed to be run in Google Colab (in the spirit of reproducibility!) please feel free to clone the repo and updload it to your drive if you want to try it out. Note that the model and dataset choices are a lot simpler than the original paper's, this is mostly due to the resource constraint of running on Colab. It was a fun excercise to reproduce similar results on a total different architecture and dataset! 

The order the notebooks should run generally follows what I did in implementing the paper, namely:

1. The first notebook trains a simple network on the fashion mnist dataset naturally and adversarailly, identifying unfairness in the adversarial training process. 

2. The second notebook implements the authors' first pass FRL algorithm for reweighing each class by their separate natural and boundary errors (see paper for details.)

3. The third notebook implements the authors' second pass algorithm for remargining the epsilon of each class' adversarial examples based on their boundary errors (see paper for details.)

Most everything should be able to run without a Colab+ subscription. 
