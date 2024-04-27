# Alignment Regularization for Sparse Autoencoders

This repository contains an implementation of 'alignment regularization' (AR) for sparse autoencoders (SAEs). AR consists of training multiple SAEs in parallel and penalizing for low cosine similarity in their decoder weights, hidden states or reconstructions in training. The central motivation is to encourage the SAEs to learn a common representation. I expect this to improve the quality of the SAEs because features more likely to be discovered by many SAEs may be more likely to be features of the input than features found by a single SAE.

This repository is currently messy, and only partially supports AR and various benchmarks to test it (these benchmarks may later be spun out into an SAE benchmarking package). The supported experiments are detailed below. Basic configurations are provided for each experiment.

## Recovering Known Ground Truth Features

In this experiment, SAEs are trained on synthetic data with known ground truth features designed to mimic feature superposition in neural networks. This is a convenient testbed for measuring how well an SAE is approximating ground truth features, and is ideal for quickly testing AR implementations. This experiment can be run with the command:
```
python main.py --config configs/synthetic_config.yaml
```

## Finding Features in GPT2-Small

This is a common task to use an SAE for. An LLM is prompted to explain what feature dictionary elements might correspond to, and the efficacy of that explanation is gauged based on how accurately an LLM can predict the activation of the neuron corresponding to that dictionary element. This can be treated as a proxy for the monosemanticity of those dictionary elements. This experiment can be run with the command:
```
python main.py --config configs/gpt2_config.yaml
```

## Denoising Multivariate Time Series Data

In this experiment we train a transformer to predict time series data as denoised by an SAE. If a transformer is able to more easily predict data having been passed through an SAE, it could indicate that the SAE has successfully denoised the raw data. We use EEG data for this experiment. This experiment can be run with the command:
```
python main.py --config configs/eeg_config.yaml
```

## SAE Training on GPT-2 Activations

This experient trains an SAE (both with or without AR) on the activations of GPT2-Small (and should generalize to all GPT-2 series models). It also includes code for creating a dataset of activations (MLP outputs by default). This experiment can be run with the command:
```
python main.py --config configs/gpt2_sae_config.yaml
```
