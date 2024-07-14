# Mutual Regularization for Sparse Autoencoders

This repository contains an implementation of 'mutual regularization' (MR) for sparse autoencoders (SAEs). MR consists of training multiple SAEs in parallel and penalizing for low cosine similarity in their decoder weights, hidden states or reconstructions in training. The central motivation is to encourage the SAEs to learn a common representation. I expect this to improve the quality of the SAEs because features more likely to be discovered by many SAEs may be more likely to be features of the input than features found by a single SAE.

This repository is currently messy, and only partially supports MR and various benchmarks to test it (these benchmarks may later be spun out into an SAE benchmarking package). The supported experiments are detailed below. Basic configurations are provided for each experiment.

## Recovering Known Ground Truth Features

In this experiment, SAEs are trained on synthetic data with known ground truth features designed to mimic feature superposition in neural networks. This is a convenient testbed for measuring how well an SAE is approximating ground truth features, and is ideal for quickly testing MR implementations. This experiment can be run with the command:
```
python main.py --config configs/synthetic_config.yaml
```
