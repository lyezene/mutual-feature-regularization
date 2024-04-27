from openai import OpenAI


class GPT4Helper:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model_engine = "gpt-4-turbo"

    def explain_feature(self, feature_index, activation_str):
        messages = [
            {
                "role": "system",
                "content": "We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words. The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.",
            },
            {
                "role": "user",
                "content": f"Feature {feature_index} Top Activations:\n{activation_str}",
            },
        ]
        return (
            self.client.chat.completions.create(
                messages=messages, model=self.model_engine, temperature=0
            )
            .choices[0]
            .message.content.strip()
        )

    def predict_activations(self, feature_description, tokens):
        messages = [
            {
                "role": "system",
                "content": "Given the description of a neuron in a language model and a list of tokens, predict the normalized (0-10) and discretized (to whole numbers) activations for each token. Assume the description accurately reflects the feature's behavior. The activation format is token<tab>activation value.",
            },
            {
                "role": "user",
                "content": f"Feature Description: {feature_description}\nTokens: {', '.join(tokens)}\nPredict the activations for each token based on the feature description:",
            },
        ]
        chat_completion = self.client.chat.completions.create(
            messages=messages, model=self.model_engine, temperature=0
        )
        return chat_completion.choices[0].message.content.strip()
