import os
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

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
    arr = arr.numpy()
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
        self.conv3d1 = nn.Linear(8, 8)
        self.tanh1 = nn.Tanh()
        self.conv3d2 = nn.Linear(8, 8)
        self.tanh2 = nn.Tanh()
        self.conv3d3 = nn.Linear(8, 8)
        self.tanh3 = nn.Tanh()
        self.conv3d4 = nn.Linear(8, 8)
        self.tanh4 = nn.Tanh()

        self.dense = nn.Linear(8, 8)

    def forward(self, x):
        x = x.float()  # Cast the input tensor to float32

        x = self.tanh1(self.conv3d1(x))
        x = self.tanh2(self.conv3d2(x))
        x = self.tanh3(self.conv3d3(x))
        x = self.tanh4(self.conv3d4(x))

        x = self.dense(x)

        return x


# Prepare the dataset
class LLMDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        size = cfg["sequenceSize"]
        sample = self.dataset[index]
        xs = sample[:size]
        ys = sample[size : size + size]
        assert len(xs) == len(ys) == size
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(
            ys, dtype=torch.float32
        )


# Load and preprocess the dataset
def load_dataset(books):
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


def main():
    books = sorted(
        [
            os.path.join("/content/simplebooks/simplebooks-2", f)
            for f in os.listdir("/content/simplebooks/simplebooks-2")
        ]
    )
    dataset = load_dataset(books)

    # Create data loader
    dataloader = data.DataLoader(
        LLMDataset(dataset), batch_size=cfg["batchSize"], shuffle=True
    )

    # Initialize the model, loss function, and optimizer
    model = LLM()
    criterion = nn.MSELoss()  # Changed loss function to Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for batch_xs, batch_ys in tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            optimizer.zero_grad()

            outputs = model(batch_xs)
            loss = criterion(outputs, batch_ys)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (
                (torch.argmax(outputs, dim=-1) == torch.argmax(batch_ys, dim=-1))
                .float()
                .mean()
                .item()
            )

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

        # Save the model checkpoint
        torch.save(model.state_dict(), "./model.pth")

        # Generate sample output
        with torch.no_grad():
            random_index = np.random.randint(len(dataset))
            input_seq = dataset[random_index]
            input_tensor = torch.tensor(input_seq).unsqueeze(0)

            output_tensor = model(input_tensor)
            output_seq = reparse(output_tensor[0])

            print(
                "---------------------------------INPUT-----------------------------------------"
            )
            print(" ".join(reverser(torch.tensor(word)) for word in input_seq))
            print(
                "--------------------------------PREDICT----------------------------------------"
            )
            print(output_seq)


if __name__ == "__main__":
    main()
