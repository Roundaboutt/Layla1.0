import torch
from torch import nn


class net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32), # <--- 添加 BatchNorm2d
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32), # <--- 添加 BatchNorm2d
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64), # <--- 添加 BatchNorm2d
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(1024, 64),
            nn.BatchNorm1d(64), # <--- 添加 BatchNorm1d
            nn.ReLU(),
            # 可以在这里添加 nn.Dropout(0.5) 试试看效果

            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.module(input)
        return output
    
if __name__ == '__main__':
    test_net = net()
    input = torch.ones((64,3,32,32))
    output = test_net(input)
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)
    print(test_net) # 打印模型结构，可以看到 BatchNorm 层
    total_params = sum(p.numel() for p in test_net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

