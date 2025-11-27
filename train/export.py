import torch
from torch import nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
net = Net(input_size=784, hidden_size=500, num_classes=10)
net.load_state_dict(torch.load("mnist_net.pth"))

with open("model.bin", "wb") as f:
    # 转换为 numpy 数组后，强制指定类型为 float32
    fc1_weight = net.fc1.weight.detach().cpu().numpy().astype(np.float32)
    fc1_bias = net.fc1.bias.detach().cpu().numpy().astype(np.float32)
    fc2_weight = net.fc2.weight.detach().cpu().numpy().astype(np.float32)
    fc2_bias = net.fc2.bias.detach().cpu().numpy().astype(np.float32)

    print(f"Writing fc1.weight with shape {fc1_weight.shape} and dtype {fc1_weight.dtype}") # 应该输出 float32
    f.write(fc1_weight.tobytes())

    print(f"Writing fc1.bias with shape {fc1_bias.shape} and dtype {fc1_bias.dtype}")
    f.write(fc1_bias.tobytes())

    print(f"Writing fc2.weight with shape {fc2_weight.shape} and dtype {fc2_weight.dtype}")
    f.write(fc2_weight.tobytes())

    print(f"Writing fc2.bias with shape {fc2_bias.shape} and dtype {fc2_bias.dtype}")
    f.write(fc2_bias.tobytes())

print("模型权重已成功导出到 model.bin")