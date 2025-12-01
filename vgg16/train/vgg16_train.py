import torch
import torchvision
from torch import nn
from vgg16_model import net
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import time


# CIFAR-10 训练集的均值和标准差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 随机裁剪，先pad 4像素，再裁剪回32x32
    transforms.RandomHorizontalFlip(),    # 随机水平翻转
    transforms.ToTensor(),                # 转换为 Tensor，并归一化到 [0, 1]
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD) # 标准化
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD) # 测试集也必须标准化
])

train_dataset = torchvision.datasets.CIFAR10(
    "/home/a1097/Project/layla/vgg16/data/train_dataset",
    train=True,
    transform=train_transform, # 使用新的 train_transform
    download=True
)

test_dataset  =torchvision.datasets.CIFAR10(
    "/home/a1097/Project/layla/vgg16/data/test_dataset",
    train=False,
    transform=test_transform, # 使用新的 test_transform
    download=True
)


test_dataset_size = len(test_dataset)
train_dataset_size = len(train_dataset)

#加载到dataloader
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=64)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64)

device = torch.device("cuda")


#创建网络模型
Net = net().to(device)

#损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

writer = SummaryWriter("./vgg16/log_trains")


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

Net.apply(init_weights)

epoch = 50  #训练周期
total_train_step = 0    #训练次数
total_test_step = 0     #测试次数

optimizer = torch.optim.Adam(Net.parameters(),lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoch)

#计时
start_time = time.time()

#训练
for i in range(epoch):
    print(f'--------第{i + 1}轮训练开始--------')
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = Net(imgs)
        loss = loss_fn(output,targets)

        #用优化器来优化参数
        optimizer.zero_grad()   #梯度清零
        loss.backward()         #对损失函数求梯度
        optimizer.step()        #优化参数

        total_train_step += 1

    #每一轮训练完之后进行测试
    total_test_loss = 0         #测试中总共的损失
    total_test_accuracy = 0     #总共的精度(预测正确的个数)
    scheduler.step()

    #不求梯度
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = Net(imgs)
            loss = loss_fn(output,targets)
            total_test_loss += loss
            accuracy = (output.argmax(1) == targets).sum()
            total_test_accuracy += accuracy
    print(f'整体测试集上的loss:{total_test_loss.item():.2f}')
    print(f'整体测试集上的正确率:{(total_test_accuracy/test_dataset_size) * 100:.2f}%')
    writer.add_scalar("test",loss.item(),total_train_step)
    writer.add_scalar("test_accuracy",total_test_accuracy/test_dataset_size,total_train_step)
    total_train_step += 1

torch.save(Net.state_dict(),"./vgg16/vgg16.pth")
print('模型已保存')