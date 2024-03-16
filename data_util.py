import numpy as np
import torch
from torchtyping import TensorType
from typing import Iterable
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import numpy as np
import torch
from torchtyping import TensorType
from typing import Iterable
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class ArticleDataset():
    """
    Dataset class for text summarization.
    
    Args:
        data (dict): Dictionary containing the data, typically with 'train' and 'test' keys.
        tokenizer: Tokenizer object for tokenizing the data.
    
    Attributes:
        article_input_ids: Tokenized input article data.
        highlights_input_ids: Tokenized input summary data.
    """

    def __init__(self, data , tokenizer):
        self.article_input_ids = tokenizer([item for item in data['train']['article']]).input_ids
        self.highlights_input_ids = tokenizer([item for item in data['train']['highlights']]).input_ids

    def __getitem__(self, ix):
        return self.prompts_input_ids[ix]

    def __len__(self):
        return len(self.prompts_input_ids)
    

class TextSummarizationDataGenerator():
    """
    Data generator for text summarization task.
    
    Args:
        text_data: Input text data.
        summary_data: Target summary data.
        batch_size (int): Batch size for generating data batches.
    
    Attributes:
        text_data: Input text data.
        summary_data: Target summary data.
        batch_size: Batch size for generating data batches.
        num_samples: Total number of samples in the dataset.
        indices: Indices of the samples.
    """
    def __init__(self, text_data, summary_data, batch_size):
        self.text_data = text_data
        self.summary_data = summary_data
        self.batch_size = batch_size
        self.num_samples = len(text_data)

    def __iter__(self):
        self.indices = np.arange(self.num_samples)
        return self

    def __next__(self):
        if len(self.indices) >= self.batch_size:
            sampled_indices = np.random.choice(self.indices, self.batch_size, replace=False)
            text_batch = [self.text_data[i] for i in sampled_indices]
            summary_batch = [self.summary_data[i] for i in sampled_indices]
            self.indices = np.delete(self.indices, sampled_indices)
            return text_batch, summary_batch
        else:
            raise StopIteration




@dataclass
class PPORLElement:
    """
    Dataclass for a single element in PPO RL training.
    """
    article_tensor: TensorType["article_size"]
    summary_tensor: TensorType["summary_size"]
    logprobs: TensorType["summary_size", "vocab_size"]
    values: TensorType["summary_size"]
    rewards: TensorType["summary_size"]


@dataclass
class PPORLBatch:
    """
    Dataclass for a batch of elements in PPO RL training.
    """
    article_tensor: TensorType["batch_size", "article_size"]
    summary_tensor: TensorType["batch_size", "summary_size"]
    logprobs: TensorType["batch_size", "summary_size", "vocab_size"]
    values: TensorType["batch_size", "summary_size"]
    rewards: TensorType["batch_size", "summary_size"]



class PPORolloutStorage():
    """
    Storage class for storing rollout data in Proximal Policy Optimization (PPO) training.

    Args:
        tokenizer: Tokenizer object used to tokenize the data.

    Attributes:
        pad_token_id: ID of the padding token used for padding sequences.
        history: Iterable containing the rollout data elements.
    """

    def __init__(self, tokenizer):
        """
        Initializes the PPORolloutStorage.

        Args:
            tokenizer: Tokenizer object used to tokenize the data.
        """
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        """
        Pushes a batch of rollout elements into the storage.

        Args:
            exps (Iterable[PPORLElement]): Batch of rollout elements to push into the storage.
        """
        self.history += exps

    def clear_history(self):
        """
        Clears the rollout history.
        """
        self.history = []

    def __getitem__(self, index: int) -> PPORLElement:
        """
        Retrieves a rollout element from the storage by index.

        Args:
            index (int): Index of the element to retrieve.

        Returns:
            PPORLElement: Rollout element at the specified index.
        """
        return self.history[index]

    def __len__(self) -> int:
        """
        Returns the number of rollout elements in the storage.

        Returns:
            int: Number of rollout elements in the storage.
        """
        return len(self.history)

    def create_loader(self, mini_batch_size: int, shuffle: bool) -> DataLoader:
        """
        Creates a DataLoader to iterate over the rollout data.

        Args:
            mini_batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data during training.

        Returns:
            DataLoader: DataLoader object for iterating over the rollout data.
        """
        def collate_fn(elems: Iterable[PPORLElement]):
            """
            Collate function for padding and batching rollout elements.

            Args:
                elems (Iterable[PPORLElement]): Iterable of rollout elements.

            Returns:
                PPORLBatch: Batched and padded rollout data.
            """
            return PPORLBatch(
                pad_sequence(
                    [elem.article_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.summary_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.values for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True
                ),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
            )

        return DataLoader(self, mini_batch_size, shuffle=shuffle, collate_fn=collate_fn)
