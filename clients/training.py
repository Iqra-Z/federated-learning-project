import torch
import torch.nn.functional as F

def train_local(model, train_loader, epochs=1, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()

    return model
