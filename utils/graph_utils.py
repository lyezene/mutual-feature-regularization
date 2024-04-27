import matplotlib as plt
import seaborn as sns


class GraphingHelper:
    def __init__(self):
        pass

    @staticmethod
    def plot_results(results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        for params, losses, mmcs_scores, sim_matrices in results:
            label = f"LR={params['learning_rate']}, L1={params['l1_coef']}"

            ax1.plot(losses, label=label + " Loss", marker='o')
            ax1.set_title('Loss Across Different Configurations')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            ax2.plot(mmcs_scores, label=label + " MMCS", marker='x')
            ax2.set_title('MMCS Across Different Configurations')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean Maximum Cosine Similarity')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cosine_similarity_heatmap(similarity_matrix, params):
        plt.figure(figsize=(10, 8))
        similarity_matrix = similarity_matrix[0].cpu().numpy()
        sns.heatmap(similarity_matrix, annot=False, cmap='viridis')
        plt.title(f'Cosine Similarity Heatmap | Params: {params}')
        plt.xlabel('Ground Truth Features')
        plt.ylabel('Learned Features')
        plt.show()

    @staticmethod
    def plot_max_similarity_distribution(similarity_matrix, params):
        plt.figure(figsize=(10, 6))
        similarity_matrix = similarity_matrix[0].cpu().numpy()
        max_similarities = np.max(similarity_matrix, axis=0)
        plt.hist(max_similarities, bins=20, color='blue', alpha=0.7)
        plt.title(f'Distribution of Maximum Cosine Similarities | Params: {params}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.show()
