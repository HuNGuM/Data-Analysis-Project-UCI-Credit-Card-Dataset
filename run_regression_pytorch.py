import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time

def load_data():
    df = pd.read_csv("UCI_Credit_Card.csv")
    df.columns = df.columns.str.strip()

    X = df[["PAY_AMT1"]].to_numpy()
    y = df["BILL_AMT1"].to_numpy()

    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=64, shuffle=True)

def train_model(device):
    dataloader = load_data()

    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    start = time.time()

    for epoch in range(10):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    end = time.time()
    print(f"Training time on {device}: {end - start:.2f} seconds")

def main():
    print("\nCPU Training")
    train_model(torch.device("cpu"))

    if torch.cuda.is_available():
        print("\nGPU Training")
        train_model(torch.device("cuda"))
    else:
        print("\nGPU not available")

if __name__ == "__main__":
    main()