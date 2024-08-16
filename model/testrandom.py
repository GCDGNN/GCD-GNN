import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# 设置随机种子函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 定义包含Dropout和BatchNorm层的模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 准备数据集和DataLoader
from torch.utils.data import DataLoader, TensorDataset

seed = 42
set_seed(seed)

dataset = TensorDataset(torch.randn(100, 3), torch.randint(0, 2, (100,)))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, worker_init_fn=set_seed)

# 设置随机种子

# 初始化模型、损失函数和优化器
model = Net()
model.apply(init_weights)  # 初始化权重
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for inputs, targets in dataloader:
        set_seed(seed)  # 在每个batch开始前设置随机种子，确保Dropout一致
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 评估模型
model.eval()
with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets.float().unsqueeze(1))
        print(f"Eval Loss: {loss.item()}")
