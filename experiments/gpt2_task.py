from data.gpt2_dataset import GPT2ActivationDataset
from utils.gpt2_utils import process_activations, get_feature_dict, predict_and_evaluate, stream_data
from utils.gpt4_utils import GPT4Helper

def run(device, config):
    activation_dataset = GPT2ActivationDataset("gpt2", device)
    gpt4_helper = GPT4Helper(config['gpt2_api_key'])

    data_stream = stream_data()

    all_data = []
    for entry in data_stream:
        input_text = entry["text"]
        if not input_text.strip():
            continue

        activations, tokens = activation_dataset(input_text)
        processed_activations = process_activations(activations)
        all_data.append((processed_activations, tokens))

    feature_explanations = get_feature_dict(gpt4_helper, all_data)
    rho_scores = predict_and_evaluate(gpt4_helper, feature_explanations, all_data)

    return feature_explanations, rho_scores
