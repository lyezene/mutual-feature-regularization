import torch
from transformers import GPT2Model, GPT2Config


class GPT2Shortcut(torch.nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = GPT2Model(config).h[0].ln_1
        self.attn = GPT2Model(config).h[0].attn
        self.ln_2 = GPT2Model(config).h[0].ln_2
        self.mlp = GPT2Model(config).h[0].mlp

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask=attention_mask)[0]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        return mlp_output
