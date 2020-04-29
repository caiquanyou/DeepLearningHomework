import torch
import torch.nn as nn
import torch.optim as Opt
import torchvision
from torchvision import datasets,transforms
import cv2
import matplotlib.pyplot as plt
# 定义是用CPU还是GPU
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 定义批次大小
b_size = 64
# 学习率
learning_rate = 0.001
# 由于是多分类问题，采用交叉熵作为损失函数
loss_func = nn.CrossEntropyLoss()

#数据集下载（看教程的时候看到torch官方可以用dataloader下载mnist数据集）
#0.1037和0.3081是数据集的均值和方差，分别减去均值除以方差处理得到数据的正则化
train_data = torch.utils.data.DataLoader(
    datasets.MNIST('mnist_data', train = True, download = True,
              transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1037,), (0.3081,))
              ])),batch_size = b_size, shuffle = True)
#设置测试集
test_data = torch.utils.data.DataLoader(
    datasets.MNIST('mnist_data', train = False, transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.1037,), (0.3081,))
    ])),batch_size = b_size, shuffle = True)
'''
#查看数据
images, labels = next(iter(train_data))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print(labels)
cv2.imshow('win', img)
key_pressed = cv2.waitKey(0)
'''

#构建网络框架，这里继承nn的Module
class Nerualnetwork_hw_1 (nn.Module):
    def __init__(self):
        super().__init__()
    #设置卷积层1(输入为单通道的图像数据大小为1*28*28)
    #设置了16个3*3的卷积核，步长为1，padding为2，输出为16*30*30，进行ReLU激活函数
    #经过pooling层后输出为16*15*15
        self.conv1 = nn.Sequential(
        nn.Conv2d(1,16,3,1,2),
        nn.ReLU(True),
        nn.MaxPool2d(2,2)
        )
    #经过32个卷积核后，输出为32*12*12，经过pooling层后为32*6*6
        self.conv2 = nn.Sequential(
        nn.Conv2d(16,32,4,1),
        nn.ReLU(True),
        nn.MaxPool2d(2,2)
        )
    #全连接层,最后输出为10
        self.fullconnect = nn.Sequential(
        nn.Linear(32*6*6,160),
        nn.Linear(160,10)
        )
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        #拉直操作，输出为1*32*6*6
        x_3 = x_2.view(x_2.size(0),-1)
        x_4 = self.fullconnect(x_3)
        return x_4

#模型训练
def model_train(times,model,device,loss_function,optimizer,train_data):
    model.train()
    for i,data in enumerate(train_data):
        image,label = data
        image,label = image.to(device), label.to(device)
        optimizer.zero_grad()
        trainning = model(image)
        loss = loss_function(trainning,label)
        loss.backward()
        optimizer.step()
        if i % 200 == 0 :
            print('Train Times: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            times+1, i * len(image), len(train_data.dataset),
            100. * i / len(train_data), loss.item()))
#模型测试
def model_test(model,device,test_data,loss_function):
    model.eval()
    test_total = 0
    test_correct = 0
    with torch.no_grad():
        for image,label in test_data:
            image,label = image.to(device), label.to(device)
            testing = model(image)
            loss = loss_function(testing,label)
            #计算损失和
            test_total += label.size(0)
            _, pred = torch.max(testing, 1)
            test_correct += (pred == label).sum().item()
    print('Correct rate is : {:.6f} %,Loss is \t{:.6f}'.format((test_correct/test_total*100), loss.item()))
    result.append(test_correct/test_total*100)

#训练模型
if __name__ == '__main__':
    #在GPU或者CPU运行
    run_model = Nerualnetwork_hw_1().to(DEVICE)
    # 优化器设置为随机梯度下降算法
    optimizer = Opt.SGD(run_model.parameters(), lr=learning_rate, momentum=0.9)
    r_times = 12
    result = []
    for times in range(r_times):
        model_train(times, run_model, DEVICE, loss_func, optimizer, train_data)
        model_test(run_model, DEVICE, test_data, loss_func)
    model_test(run_model, DEVICE, test_data, loss_func)
    torch.save(run_model.state_dict(), 'model_parameters.pkl')
    plt.plot(range(r_times+1), result)
    plt.show()