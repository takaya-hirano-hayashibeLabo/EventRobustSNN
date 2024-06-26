import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tonic.datasets import NMNIST
from tonic.transforms import ToFrame,Denoise
from tonic import transforms

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(34 * 34, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 34 * 34)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the N-MNIST dataset
sensor_size=NMNIST.sensor_size
transform = transforms.Compose([
    Denoise(filter_time=10000), #filter_timeの間イベントがなければそのフレームは無視される
    ToFrame(sensor_size=sensor_size, time_window=1000), #ここで、ある時間windowごとのフレームデータに変換される
])

trainset = NMNIST(save_to='./data', transform=transform, train=True)
testset =  NMNIST(save_to='./data', transform=transform, train=False)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        print(inputs.shape)
        print(labels.shape)
        break
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')