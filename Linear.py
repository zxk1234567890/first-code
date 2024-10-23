#线性回归从零开始
import random
import torch
#from d2l import torch as d2l

#生成一个合成数据集
def synthetic_data(w,b,num_examples):
    '''生成y=Xw+b+噪声'''
    X = torch.normal(0,1,(num_examples,len(w)))  # 生成一个张量维度（num_examples,len(w)）,均服从均值为0，标准差为1的正太分布
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))  #y.reshape(-1,1)转成列向量，只有一列

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])
# 读取数据集
#用于生成大小为batch-size的小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        yield features[batch_indices],labels[batch_indices]

#初始化模型参数
#w = torch.randn(2,1)
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
print(f'w={w}')
b = torch.zeros(1, requires_grad=True)
#定义模型
def linreg(X,w,b):
    return torch.matmul(X,w)+b

#定义损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

#定义优化算法

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param.data -= lr*param.grad/batch_size
            param.grad.zero_()

#定义超参数
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10
#开始训练
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features, labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')
