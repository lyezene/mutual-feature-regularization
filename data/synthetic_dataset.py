import torch
from torch.utils.data import IterableDataset
from huggingface_hub import hf_hub_download
import os


class SyntheticIterableDataset(IterableDataset):
    def __init__(self, repo_id, cache_dir=None, chunk_size=1000, num_epochs=1):
        self.repo_id = repo_id
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'hf_cache')
        self.batch_files = [f'batch_{i}.pt' for i in range(0, 1000000000, 50000000)]
        self.total_samples = 1000000000
        self.chunk_size = chunk_size
        self.num_epochs = num_epochs
        self.reset()

    def reset(self):
        self.current_epoch = 0
        self.current_file_index = 0
        self.current_batch = None
        self.batch_index = 0
        self.current_chunk_start = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_epoch >= self.num_epochs:
            raise StopIteration

        if self.current_batch is None or self.batch_index >= len(self.current_batch):
            if self.current_file_index >= len(self.batch_files):
                self.current_epoch += 1
                if self.current_epoch >= self.num_epochs:
                    raise StopIteration
                self.current_file_index = 0
                self.current_chunk_start = 0

            if self.current_chunk_start >= 50000000:  # Move to next file
                self.current_file_index += 1
                self.current_chunk_start = 0

            if self.current_file_index >= len(self.batch_files):
                self.current_epoch += 1
                if self.current_epoch >= self.num_epochs:
                    raise StopIteration
                self.current_file_index = 0
                self.current_chunk_start = 0

            file_path = hf_hub_download(self.repo_id, f"data/{self.batch_files[self.current_file_index]}", 
                                        repo_type="dataset", cache_dir=self.cache_dir)
            
            full_batch = torch.load(file_path)
            self.current_batch = full_batch[self.current_chunk_start:self.current_chunk_start + self.chunk_size]
            self.current_chunk_start += self.chunk_size
            self.batch_index = 0

        item = self.current_batch[self.batch_index].float()
        self.batch_index += 1
        return (item,)

    def __len__(self):
        return self.total_samples * self.num_epochs
