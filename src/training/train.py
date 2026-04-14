# Training script for image-to-caption classification model.
# This script:
# 1. Loads the dataset (images + captions)
# 2. Trains a CNN using cross-entropy loss
# 3. Updates model weights using Adam optimizer
# 4. Saves the trained model and label mapping for inference
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json
import sys
import os
sys.path.append(os.path.abspath("."))

from dataset import ImageCaptionDataset
from first_model import SimpleCNN

def train():
    # loading data
    dataset = ImageCaptionDataset("data/mock/data.jsonl")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SimpleCNN(num_classes=len(dataset.label_map))
   # Define optimizer and loss function (CrossEntropy for classification)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        total_loss = 0
        # forward pass
        for imgs, labels in loader:
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            # back propogation
            optimizer.zero_grad()
            loss.backward()
            # updating weights
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss}")

    # Save model
    torch.save(model.state_dict(), "model.pth")

    # Save label map
    with open("label_map.json", "w") as f:
        json.dump(dataset.label_map, f)

    print("Model + label map saved!")

if __name__ == "__main__":
    train()
