from datasets import *
import utils
from model import Net

transform = transforms.Compose([utils.ToTensor()])
test_dataset = MNIST_Datasets('testing', './data', transform)
testloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print 'Device:', device

model = Net()
model.load_state_dict(torch.load('./trained_model/checkpoint.pth.tar'))
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