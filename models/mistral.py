import torch
import torch.nn as nn
from transformers import MistralConfig, MistralModel

class MistralForEEGPrediction(nn.Module):
    def __init__(self, config):
        super(MistralForEEGPrediction, self).__init__()
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.input_layer = (
            nn.Linear(config.num_channels, config.hidden_size)
            if config.hidden_size > config.num_channels
            else nn.Identity()
        )
        self.model = MistralModel(config)
        self.output_layer = (
            nn.Linear(max(config.hidden_size, config.num_channels), config.num_channels)
            if config.hidden_size > config.num_channels
            else nn.Identity()
        )

    def forward(self, x):
        x = self.input_layer(x)
        outputs = self.model(inputs_embeds=x.float()).last_hidden_state
        outputs = self.output_layer(outputs)
        return outputs
