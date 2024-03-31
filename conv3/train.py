import os
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


import lightning as L


DATAPATH = pathlib.Path("/datasets/simplebooks/")

# Configuration
cfg = {
    "sequenceSize": 512,
    "predictSteps": 512,
    "dimension": 2048,
    "batchSize": 16,
    "arrayDimension": 8,
}


# Load and preprocess the word embeddings
def load_embeddings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:  # Added encoding parameter
        lines = f.readlines()[1:]
    w2v = {}
    for line in lines:
        tokens = line.strip().split(" ")
        key = tokens[0]
        vec = np.array(tokens[1:], dtype=np.float32)
        w2v[key] = vec
    return w2v


w2v = load_embeddings(DATAPATH / "./vec.vec")
vw2v = list(w2v.items())


def convert(n):
    return w2v.get(n, np.zeros(8))


def reverser(arr):
    arr = arr.cpu().numpy()
    best = float("inf")
    max_word = None
    for word, vec in vw2v:
        if len(vec) == 0:
            vec = np.zeros(8)
        b = euclidean_distances([vec], [arr])[0][0]
        if b < best:
            best = b
            max_word = word
    return max_word


def reparse(v):
    v = [reverser(s) if s.nelement() else s for s in v]
    return " ".join(v)


# Define the model architecture
class LLM(nn.Module):
    def __init__(self):
        super(LLM, self).__init__()
        self.conv3d1 = nn.Conv3d(
            8, 32, kernel_size=1, padding="same"
        )  # Adjust padding if needed
        self.conv3d2 = nn.Conv3d(
            32, 32, kernel_size=5, padding="same"
        )  # Consider padding for even filters
        self.conv3d3 = nn.Conv3d(32, 32, kernel_size=1, padding="same")
        self.conv3d4 = nn.Conv3d(
            32, 32, kernel_size=8, padding="same"
        )  # Consider padding for odd filters
        self.dense1 = nn.Linear(32, 8)  # Efficient matrix multiplication

    def forward(self, x):
        x = x.reshape(-1, 8, 8, 8, 8)
        x = self.conv3d1(x)
        x = nn.functional.tanh(x)  # Add activation function after Conv3D1
        x = self.conv3d2(x)
        x = nn.functional.tanh(x)
        x = self.conv3d3(x)
        x = nn.functional.tanh(x)
        x = self.conv3d4(x)
        x = nn.functional.tanh(x)
        x = x.reshape(-1, 512, 32)
        x = self.dense1(x)
        return x


class LLMDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        size = cfg["sequenceSize"]
        sample = self.dataset[index]
        xs = np.array(sample[:size])
        ys = np.array(sample[size : size + size])
        assert len(xs) == len(ys) == size
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(
            ys, dtype=torch.float32
        )


# Load and preprocess the dataset
def load_dataset(data_path):
    books = sorted([os.path.join(data_path, f) for f in os.listdir(data_path)])

    dataset = []
    size = cfg["sequenceSize"]
    psteps = cfg["predictSteps"]
    chunk_len = size + psteps
    for book_path in books:
        with open(book_path, "r", encoding="utf-8") as f:  # Added encoding parameter
            book = f.read().split(" ")
        book = [convert(word) for word in book]
        # here we split into chunks of sequenceSize+predictSteps so that the first half is the input and the second half is target.
        # Currently, the chunks don't overlap. TODO: is this correct?
        book_chunks = [
            book[i : i + chunk_len] for i in range(0, len(book) - chunk_len, 512)
        ]
        dataset.extend(book_chunks)
    return dataset


class BookDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: pathlib.Path = DATAPATH / "simplebooks" / "simplebooks-2",
        batch_size: int = 512,
        num_workers: int = 4,
        validation_size: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_size = validation_size

    def setup(self, stage: str) -> None:
        self.dataset = load_dataset(self.data_dir)

    def train_dataloader(self):
        return data.DataLoader(
            LLMDataset(self.dataset[: self.validation_size]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            LLMDataset(self.dataset[-self.validation_size :]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class Conv3Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = LLM()
        print("Model = ", self.model)

    def compute_loss(self, batch):
        batch_xs, batch_ys = batch
        criterion = nn.MSELoss()  # Changed loss function to Mean Squared Error
        outputs = self.model(batch_xs)
        accuracy = (
            (torch.argmax(outputs, dim=-1) == torch.argmax(batch_ys, dim=-1))
            .float()
            .mean()
            .item()
        )

        return criterion(outputs, batch_ys), accuracy, outputs

    def training_step(self, batch, batch_idx):
        print("Training step")

        loss, accuracy, _ = self.compute_loss(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        print("Validation step")
        loss, accuracy, outputs = self.compute_loss(batch)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, logger=True)
        if batch_idx >= 2:
            return
        print("Reparsing")
        output_seq = reparse(outputs[0][0:50])
        print("Reparsing done")

        print(
            "---------------------------------INPUT-----------------------------------------"
        )
        # print(" ".join(reverser(torch.tensor(word)) for word in input_seq))
        print(
            "--------------------------------PREDICT----------------------------------------"
        )
        print(output_seq)
        print("Validation done")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
