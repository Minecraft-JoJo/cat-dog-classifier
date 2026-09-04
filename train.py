from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchinfo import summary
from config import setting
from tqdm import tqdm
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

import copy
import time




def data_process(cfg):

    transforming_train =  transforms.Compose([
        transforms.Resize((cfg.input_size,cfg.input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                             std=[0.229, 0.224, 0.225])])

    transforming_val =  transforms.Compose([
            transforms.Resize((cfg.input_size,cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                             std=[0.229, 0.224, 0.225])])


    data_train = ImageFolder(root=cfg.data_root,transform=transforming_train)
    data_val = ImageFolder(root=cfg.val_root,transform=transforming_val)
    # 会自动根据文件夹名字编号
    

    x= Data.DataLoader(dataset=data_train,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,pin_memory=True)
    x_val = Data.DataLoader(dataset=data_val,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True)
    return x,x_val



def train_process(model,num_epochs,x,x_val):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    print('硬件:',device)

    best_acc = 0   
    best_wts = copy.deepcopy(model.state_dict())
    

    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []

    start= time.time() 

    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    try:
        for epoch in range(num_epochs):

            start_time = time.time()  #时间戳
            print("训练轮次:",epoch+1)
            print('----------------------')



            each_train_loss_total = 0
            each_train_acc_total = 0

            each_val_loss_total = 0
            each_val_acc_total = 0

            train_num = 0
            val_num = 0

            model.train()#开启训练模式

            train_pbar = tqdm(x, total=len(x), desc=f"Epoch {epoch+1}/{num_epochs} Training")

            for step, (b_x,b_y) in enumerate(train_pbar):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                '''
                这时候运行就会丢弃神经元,eval模式就不会
                '''

                output = model(b_x)

                pre_lab = torch.argmax(output,dim=1)  

                '''
                output形状是[batch_size,10]
                dim=1指从第二个维度照,,也就是一行一行照,会输出batch_size个数字,数字是位置下标
                由于nn.CrossEntropyLoss()函数里面自带softmax和log计算,这里需要输入原始数据作为参数而不用传入激活函数
                MSELoss 或 NLLLoss就需要传入激活函数之后的值
                '''

                loss = criterion(output,b_y)

                optimizer.zero_grad()  #初始化梯度

                loss.backward() #计算梯度

                optimizer.step()  #更新梯度



                each_train_loss_total += loss.item() * b_x.size(0) 
                '''
                loss是这个batch的平均损失,乘b_x.size(0)是因为最后批次不够设定的数量
                loss是一个有计算图的张量,loss.item()就把数据取出来减少内存,b_x.size(0) 是现在批次真实的图片数量
                train_loss是真实的所有图片loss的总和,最终还要除以训练集的图片数量来求真实的loss
                张量里size()是方法,返回元组,输入数字选中维度
                '''
                train_num += b_x.size(0)  #同理这里是求真实的所有批次数量总和

                each_train_acc_total += torch.sum(pre_lab == b_y)  #计算每个批次里对的个数的总和

                train_pbar.set_postfix({
                'loss': loss.item(),
                'val_acc': f"{each_train_acc_total.item() / (train_num if train_num>0 else 1):.4f}"
            })


            with torch.no_grad():  #无梯度的推理模式,前向传播

                model.eval()  #评估模式,前向传播模式   这两个都会取消张量的计算图

                for step, (b_x,b_y) in enumerate(x_val):   
                    b_x = b_x.to(device)
                    b_y = b_y.to(device)

                    output = model(b_x)

                    pre_lab = torch.argmax(output,dim=1) 

                    loss = criterion(output,b_y) 

                    each_val_loss_total += loss.item() * b_x.size(0) 
                    val_num += b_x.size(0)

                    each_val_acc_total += torch.sum(pre_lab == b_y)

            loss_train.append((each_train_loss_total /train_num)) #当前轮的loss加到loss_train
            acc_train.append((each_train_acc_total/train_num).double().item())  #双精度,item适用于标量张量,不过两个张量之间计算就不用item
            print(f"train_loss为{loss_train[epoch]}")
            print(f"train_acc为{acc_train[epoch]:.4f}")

            loss_val.append((each_val_loss_total /val_num)) 
            acc_val.append((each_val_acc_total/val_num).double().item())
            print(f"val_loss为{loss_val[epoch]}")
            print(f"val_acc为{acc_val[epoch]:.4f}")

            if acc_val[epoch] > best_acc:
                maxacc_epoch = epoch+1
                best_acc = acc_val[epoch]
                best_wts = copy.deepcopy(model.state_dict())




            end_time = time.time()  
            total_time = end_time - start_time
            minutes, seconds = divmod(total_time, 60)
            print(f"本轮耗时耗时: {int(minutes)} 分 {seconds:.1f} 秒\n\n")

        model_data = pd.DataFrame(data={
            "epoch":range(1,num_epochs+1),
            "loss_train":loss_train,
            "train_acc":acc_train,
            "loss_val":loss_val,
            "val_acc":acc_val,

            })

        end = time.time()
        total_time = end - start
        minutes, seconds = divmod(total_time, 60)
        print(f"训练共耗时: {int(minutes)} 分 {seconds:.1f} 秒")
        torch.save(best_wts, cfg.save_path)  
        print(f"测试集最好的一轮是{maxacc_epoch}")
        
        return model_data,1
    
    except KeyboardInterrupt:
        print("\n⚠️ 检测到手动中断，正在保存模型...")
        # 保存当前最佳模型（如果已更新）
        torch.save(best_wts, 'interrupted_best_model.pth')
        # 也保存当前的模型状态（如果中断发生得太早，可以保留当前进度）
        torch.save(model.state_dict(), 'interrupted_current_model.pth')
        print("✅ 模型已保存，程序退出。")
        return 'no content',0  # 直接退出，不再执行后续代码


def matplot(data):
    
    plt.figure(figsize=(16,8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(1,2,1)
    plt.plot(data.epoch,data.loss_train,label='训练集的loss',linewidth=2,linestyle='-',marker='o') 
    plt.plot(data.epoch,data.loss_val,label='测试集的loss',linewidth=2,linestyle='--',marker='o') 
    plt.xlabel('轮次',fontsize=20)  #  X轴设置
    plt.ylabel('loss',fontsize=20)  # Y轴设置
    plt.legend(loc='upper right')  #折线名字展示

    plt.subplot(1,2,2)
    plt.plot(data.epoch,data.train_acc,label='训练集的acc',linewidth=2,linestyle='--',marker='o') 
    plt.plot(data.epoch,data.val_acc,label='测试集的acc',linewidth=2,linestyle='--',marker='o') 
    plt.xlabel('轮次',fontsize=20)  
    plt.ylabel('acc',fontsize=20)  
    plt.legend(loc='upper left')  

    plt.show()
  



if __name__ == '__main__':

    cfg = setting()

    train_loader, val_loader = data_process(cfg)

    model_data,run = train_process(cfg.model, num_epochs=cfg.epochs, x=train_loader, x_val=val_loader)

    print(model_data)

    if run != 0:
        matplot(model_data)
