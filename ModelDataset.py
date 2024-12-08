import torch
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    """
      A subscriptable Pytorch map-style dataset class that handles both labelled and unlabelled data
      Accessing dataset[idx] allows reading the idx-th image and its corresponding label

      Attributes
      - - - - -
      encodings: tokenizer.encodings
      labels: list of labels

    """

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])