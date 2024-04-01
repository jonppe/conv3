import os
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np


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


def convert(n):
    return w2v.get(n, np.zeros(8))


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
    def __init__(self, books, indices):
        self.books = books
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        book_index, start, end = self.indices[index]
        book = self.books[book_index]
        size = cfg["sequenceSize"]
        xs = book[start : start + size]
        ys = book[end - size : end]
        assert len(xs) == len(ys) == size
        return xs, ys


# Load and preprocess the dataset
def load_dataset(data_path):
    books = sorted([os.path.join(data_path, f) for f in os.listdir(data_path)])

    available_indices = []
    book_tensors = []
    size = cfg["sequenceSize"]
    psteps = cfg["predictSteps"]
    chunk_len = size + psteps
    for book_index, book_path in enumerate(books):
        with open(book_path, "r", encoding="utf-8") as f:  # Added encoding parameter
            book = f.read().split(" ")
        book_tensors.append(
            torch.tensor(
                np.array([convert(word) for word in book]), dtype=torch.float32
            )
        )
        available_indices.extend(
            [(book_index, i, i + chunk_len) for i in range(0, len(book) - chunk_len, 1)]
        )
    return book_tensors, available_indices


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
        self.books, self.indices = load_dataset(self.data_dir)

    def train_dataloader(self):
        return data.DataLoader(
            LLMDataset(self.books, self.indices[: -self.validation_size]),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            LLMDataset(self.books, self.indices[-self.validation_size :]),
            batch_size=self.batch_size,
        )


class Conv3Module(L.LightningModule):
    def __init__(self, data_path: pathlib.Path = DATAPATH):
        super().__init__()
        self.model = LLM()
        print("Model = ", self.model)
        vw2v = list(w2v.items())
        self.tokens = np.array([s for s, _ in vw2v])
        self.embeddings = torch.tensor(np.array([v for _, v in vw2v]))

    def setup(self, stage=None):
        super().setup(stage)
        self.embeddings = self.embeddings.to(self.device, dtype=torch.float32)

    def reparse(self, prediction: torch.Tensor, p=float("inf")):
        indices = torch.cdist(self.embeddings, prediction, p).argmin(dim=1)
        tokens = self.tokens[indices.cpu().numpy()]
        return " ".join(tokens)

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
        loss, accuracy, outputs = self.compute_loss(batch)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, logger=True)
        if batch_idx >= 2:
            return
        output_seq = self.reparse(outputs[0])

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
