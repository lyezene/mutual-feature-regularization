## Training SAEs on Synthetic Data

In this experiment, SAEs are trained on synthetic data with known ground truth features designed to mimic feature superposition in neural networks. This is a convenient testbed for measuring how well an SAE is approximating ground truth features, and is ideal for quickly testing MR implementations. This experiment can be run with the command:
```
python main.py --config configs/synthetic.yaml
```

## Training SAEs on GPT-2 Small MLP Activations

This experiment trains SAEs on activations from the MLP layers (by default just layer 0) of GPT-2 Small.
```
python main.py --config configs/gpt2.yaml
```

## Training SAEs to Denoise EEG Data

This experiment trains SAEs on EEG data from the TUH EEG corpus. It also handles downloading and formatting the data for SAE training.
```
python main.py --config configs/eeg.yaml
```
