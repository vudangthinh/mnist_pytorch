import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import *
import utils
from model import Net

transform = transforms.Compose([utils.ToTensor()])
mnist_dataset = MNIST_Datasets('testing', './data', transform)
dataloader = DataLoader(mnist_dataset, batch_size=4, shuffle=True, num_workers=4)

n_epoches = 20

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# for i in range(len(mnist_dataset)):
#     label, img = mnist_dataset[i]
#     pred = model(img)

for epoch in range(n_epoches):
    for i_batch, (label, img) in enumerate(dataloader):
        pred = model(img)
        loss = criterion(pred, label)
        print i_batch, loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

