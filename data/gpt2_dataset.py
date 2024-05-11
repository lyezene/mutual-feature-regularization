from transformers import GPT2Tokenizer, GPT2Model
import torch


class GPT2ActivationDataset:
    def __init__(self, model_name, device):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(device)
        self.activations = []

        for i, block in enumerate(self.model.h):
            block.mlp.c_proj.register_forward_hook(self.save_activation(i))

    def save_activation(self, layer_idx):
        def hook(module, input, output):
            self.activations[layer_idx] = output.detach().cpu().numpy().squeeze(0)

        return hook

    def __call__(self, text):
        input_ids = self.tokenizer(
            text, return_tensors="pt", max_length=8, truncation=True
        )["input_ids"].to(self.model.device)
        self.activations = [[] for _ in range(len(self.model.h))]
        self.model(input_ids)
        return self.activations, self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze()
        )
