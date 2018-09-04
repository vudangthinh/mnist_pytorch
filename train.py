import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import *
import utils
from model import Net

transform = transforms.Compose([utils.ToTensor()])
mnist_dataset = MNIST_Datasets('training', './data', transform)
test_dataset = MNIST_Datasets('testing', './data', transform)

dataloader = DataLoader(mnist_dataset, batch_size=4, shuffle=True, num_workers=4)
testloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

print 'Dataset size:', len(mnist_dataset)
n_epoches = 5

model = Net()
# How to init weights in model
# grad = True, False
# Normalize
# GPU
# Save model
# Transfer learning

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print 'Device:', device
model.to(device)

# for i in range(len(mnist_dataset)):
#     label, img = mnist_dataset[i]
#     pred = model(img)
model.train()
for epoch in range(n_epoches):
    running_loss = 0
    for i_batch, (label, img) in enumerate(dataloader):
        label.requires_grad_(False)
        img.requires_grad_(False)
        label = label.to(device)
        img = img.to(device)

        pred = model(img)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print i_batch, loss.item()
        running_loss += loss.item()
        if i_batch % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss / 2000))
            running_loss = 0.0


# Test
model.eval()

total = 0
correct = 0
with torch.no_grad():
    for label, img in testloader:
        label = label.to(device)
        img = img.to(device)

        outputs = model(img)

        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print 'Accuracy:', correct * 100.0 / total