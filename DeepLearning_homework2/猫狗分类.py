from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image

#参数：批处理大小、GPU、学习率
batch_size = 16
DEVICE = torch.device('cuda')
Learning_rate = 0.01

#数据加载和划分测试集和训练集：0.9做训练集0.1做测试集
#图像加载和预处理，剪裁为224*224的图片
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(root='./data/train', transform=data_transform)
#print(len(train_dataset))
train_data, vaild_data = torch.utils.data.random_split(
    train_dataset,
    [int(0.9*len(train_dataset)),
     len(train_dataset)-int(0.9*len(train_dataset))])
train_set = torch.utils.data.DataLoader(train_data,
                                        batch_size=batch_size,
                                        shuffle=True)
test_set = torch.utils.data.DataLoader(vaild_data,
                                       batch_size=batch_size,
                                       shuffle=False)
#损失函数用交叉熵
cirterion = nn.CrossEntropyLoss()
#训练
def train(model,device,dataset,optimizer,epoch):
    model.train()
    correct = 0
    for i,(x,y) in enumerate(dataset):
        x , y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1,keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = cirterion(output,y)
        loss.backward()
        optimizer.step()
    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(
        epoch,loss/batch_size,correct,batch_size*len(dataset),100*correct/(batch_size*len(dataset))))
#验证
def vaild(model,device,dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(dataset):
            x,y = x.to(device) ,y.to(device)
            output = model(x)
            loss = cirterion(output,y)
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
        print("Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(
            loss/batch_size,correct,batch_size*len(dataset),100.*correct/(batch_size*len(dataset))))
#模型网络构建
class Nerualnetwork_hw_2(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        #卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 18, 4),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(18 * 54 * 54, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 18 * 54 * 54)
        x = self.fc(x)
        return x
if __name__ == '__main__':
    #加载模型和优化器
    model = Nerualnetwork_hw_2().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr = Learning_rate, momentum = 0.09)
    for epoch in range(1,20):
        train(model,DEVICE,train_set,optimizer,epoch)
        vaild(model,DEVICE,test_set)
    torch.save(model.state_dict(), 'model2_parameters.pkl')
    '''
    #测试自己家的猫
    classes = ('cat','dog')
    model.eval()
    img = cv2.imread("D:\Python_test\DeepLearning_homework2/2.jpg")
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = test_transform(img)
    img = img.to(DEVICE)
    img = img.unsqueeze(0)
    OP = model(img)
    pre = torch.nn.functional.softmax(OP,dim=1)
    print(pre)
    value, predicted = torch.max(OP.data, 1)
    print(predicted.item())
    print(value)
    pred_class = classes[predicted.item()]
    print(pred_class)
    '''