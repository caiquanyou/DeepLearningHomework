import numpy as np
import torch as t
from torch import nn
from torch.utils.data import DataLoader
import tqdm
from torch import optim
from torchnet import meter
import torch.nn.functional as F
#构建LSTM模型
class Nerualnetwork_hw_3(nn.Module):
    def __init__(self, length, embedding_dim, hidden_dim,num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，设置词长度和维度
        self.embedding = nn.Embedding(length, embedding_dim)
        # 定义LSTM结构
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        # 全连接层
        self.linear1 = nn.Linear(hidden_dim, 256)
        self.linear2 = nn.Linear(256,length)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        #print(seq_len)
        if hidden is None:
            #初始化矩阵为0
            h_0 = input.data.new(num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # 输入序列长度 * batch
        # 输出序列长度 * batch * 向量维度
        embeds = self.embedding(input)
        # 输出hidden的大小与序列长度有关，与LSTM层数无关，具体为： 序列长度 * batch * hidden_dim
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear1(output.view(seq_len * batch_size, -1))
        output = self.linear2(output)
        output = F.relu(output)
        output = self.softmax(output)
        return output, hidden

#模型生成结果测试
def generate(model, start_words, ix2word, word2ix):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    if gpu:
        input = input.cuda()
    hidden = None
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        #小于输入长度则使用输入词语，超过长度后开始用模型生成的诗字
        if i < start_words_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        #超过输入长度后将output作为下一个input进行
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results
#藏头诗功能实现
def gen_acrostic(model, start_words, ix2word, word2ix):
    result = []
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    if gpu:
        input = input.cuda()
    #index记录用了几个预输入的词作诗
    index = 0
    pre_word = '<START>'
    hidden = None
    # 开始生成诗句
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        # 说明上个字是句末
        if pre_word in {'<START>','，','。','？','！'}:
            if index == len(start_words):
                #用完给定的藏头字，退出运行
                break
            #使用输入词作为诗开头，预测后一个字
            else:
                #使用藏头诗的字作为开头
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            #将输出作为下一个输入
            input = (input.data.new([top_index])).view(1, 1)
        result.append(w)
        pre_word = w
    return result
def train(epoch):
    #选择设备
    if gpu:
        device = t.device("cuda")
    else:
        device = t.device("cpu")
    # 获取数据
    datas = np.load("data/tang.npz")
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = t.from_numpy(data)
    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    #print(len(word2ix))
    #print(len(ix2word))
    # 定义模型
    model = Nerualnetwork_hw_3(len(word2ix),
                        embedding_dim=embedding_dim,
                        hidden_dim = hidden_dim,
                        num_layers = num_layers)
    #选择Adam优化器
    optimizer = optim.Adam(model.parameters(),lr=lr)
    #损失函数选择交叉熵
    criterion = nn.CrossEntropyLoss()
    # 转移到相应设备上
    model.to(device)
    #设置损失计算
    loss_meter = meter.AverageValueMeter()
    #进行训练
    f = open('train_result.txt','w')
    for epoch in range(epoch):
        loss_meter.reset()
        for zi,data_ in tqdm.tqdm(enumerate(dataloader)):
            #数据转为longtensor
            data_ = data_.long().transpose(1,0).contiguous()
            #数据转移到对应设备
            data_ = data_.to(device)
            optimizer.zero_grad()
            # n个句子，前n-1句作为输入，后n-1句作为输出标签
            input_,target = data_[:-1,:],data_[1:,:]
            output,_ = model(input_)
            #view（-1）拉直target计算误差损失
            loss = criterion(output,target.view(-1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            #迭代一定次数后输出结果进行检查
            if (1+zi) % time == 0:
                print("第%s轮第%s次迭代训练损失为%s"%(epoch+1, time,str(loss_meter.mean)))
                print("\n")
                f.write("第%s轮第%s次迭代训练损失为%s"%(epoch+1, time,str(loss_meter.mean)))
                f.write("\r\n")
                #输出结果测试
                '''''''''''''''
                for word in list(u"床前明月光"):
                    gen_poetry = ''.join(generate(model,word,ix2word,word2ix))
                    print(gen_poetry)
                    f.write(gen_poetry)
                    f.write("\r\n")
                    f.flush()
                    '''''''''''''''
    #保存模型
    t.save(model.state_dict(),'%s_parameter.pth'%(model_saveas))
#测试结果
def test():
    print("加载数据集")
    #同样加载数据
    datas = np.load("data/tang.npz")
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    model = Nerualnetwork_hw_3(len(ix2word),
                        embedding_dim=embedding_dim,
                        hidden_dim = hidden_dim,
                        num_layers = num_layers)
    if gpu:
        model.to(t.device('cuda'))
    model.load_state_dict(t.load(model_save, 'cuda'))
    print("成功加载唐诗数据集\n")
    while True:
            '''
            print("请输入诗歌开头")
            start_words = str(input())
            result = ''.join(generate(model, start_words, ix2word, word2ix))
            print("生成的诗句如下：%s\n" % (result))
            '''
            print("请输入句子作为藏头诗开头")
            start_words = str(input())
            result = ''.join(gen_acrostic(model, start_words, ix2word, word2ix))
            print("生成的藏头诗如下：%s\n" % (result))
    
if __name__ == '__main__':
    #设置参数
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 3  # LSTM层数
    lr = 1e-3
    gpu = True
    epoch = 50
    batch_size = 16
    time = 1000  # 每1000次输出一次损失结果
    max_gen_len = 64  # 生成诗歌最长长度
    model_save = "./tang_parameter.pth"  # 预训练模型路径
    model_saveas = './tang'  # 模型保存文件名及路径

    #train(epoch)
    test()