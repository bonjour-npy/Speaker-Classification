import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


# 定义模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleModel()

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 定义 LambdaLR 学习率调度器和学习率函数
def lr_lambda(current_step):
    return current_step


scheduler = LambdaLR(optimizer, lr_lambda)

# 训练循环
num_steps = 10
for step in range(num_steps):
    # 当前 learning rate
    current_learning_rate = optimizer.param_groups[0]['lr']
    print(f"Step {step}: Learning rate is {current_learning_rate:.6f}")

    # 更新学习率
    scheduler.step()
