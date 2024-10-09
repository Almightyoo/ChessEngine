import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from DataLoader import parse_pgn
import time
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Adapted fully connected layer
        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flattening for fully connected layers
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.tanh(x)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "KingBase2019-pgn"
    X, y = parse_pgn(path, max_games=5000)
    print(X.shape,y.shape)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()

    num_epochs = 10
    for epoch in range(num_epochs):
        all_loss = 0
        num_loss = 0
        for data, target in dataloader:
            data, target = data.to(device), target.unsqueeze(1).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            all_loss += loss.item()
            num_loss += 1

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {all_loss/num_loss:.6f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

if __name__ == "__main__":
    start=time.time()
    train()
    end=time.time()
    print("time: ",end-start)

