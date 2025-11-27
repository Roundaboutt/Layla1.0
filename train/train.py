import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
import time


trans = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=trans, download=True)
mnist_test= torchvision.datasets.MNIST(root='./data', train=False, transform=trans, download=True)

batch_size = 512
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=True)

input_size = 784
hidden_size = 500
num_classes = 10
train_data_size = 60000
test_data_size = 10000

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

device = torch.device("cuda")
lr = 1e-3
net = Net(input_size, hidden_size, num_classes).to(device)
num_epochs = 10
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr)

total_train_step = 0
total_test_step = 0
start_time = time.time()
for i in range(num_epochs):
    print(f'--------第{i + 1}轮训练开始--------')
    for data in train_iter:
        imgs,labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = net(imgs)
        loss = loss_fn(output,labels)

        #用优化器来优化参数
        optim.zero_grad()   #梯度清零
        loss.backward()         #对损失函数求梯度
        optim.step()        #优化参数

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f'GPU训练耗时{end_time - start_time :.2f}s')
            print(f'训练次数:{total_train_step},损失值loss:{loss.item() :.2f}')

    #每一轮训练完之后进行测试
    total_test_loss = 0         #测试中总共的损失
    total_test_accuracy = 0     #总共的精度(预测正确的个数)

    #不求梯度
    with torch.no_grad():
        for data in test_iter:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = net(imgs)
            loss = loss_fn(output,labels)
            total_test_loss += loss
            accuracy = (output.argmax(1) == labels).sum()
            total_test_accuracy += accuracy
    print(f'整体测试集上的loss:{total_test_loss.item():.2f}')
    print(f'整体测试集上的正确率{total_test_accuracy/test_data_size :2f}')
    total_train_step += 1

torch.save(net.state_dict(),f'mnist_net.pth')
print('模型已保存')
