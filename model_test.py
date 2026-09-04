import torch
from torch import nn
from torchinfo import summary

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,kernel_size=1,out_channels=c1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,kernel_size=1,out_channels=c2[0]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=c2[0],kernel_size=3,out_channels=c2[1],padding=1),
                    nn.ReLU(inplace=True)
                )

        self.branch3 = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels,kernel_size=1,out_channels=c3[0]),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=c3[0],kernel_size=5,out_channels=c3[1],padding=2),
                            nn.ReLU(inplace=True)
                        )
        
        self.branch4 = nn.Sequential(
                            nn.MaxPool2d(kernel_size=3,padding=1,stride=1),
                            nn.Conv2d(in_channels=in_channels,kernel_size=1,out_channels=c4),
                            nn.ReLU(inplace=True)
                                )

    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1,b2,b3,b4],dim=1)


class GoogLeNet(nn.Module,):
    def __init__(self,num_classes,in_channels):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


         # Inception 模块系列
        self.inception3a = Inception(192, 64, (96, 128), (16, 32), 32)
        self.inception3b = Inception(256, 128, (128, 192), (32, 96), 64)

        self.inception4a = Inception(480, 192, (96, 208), (16, 48), 64)
        self.inception4b = Inception(512, 160, (112, 224), (24, 64), 64)
        self.inception4c = Inception(512, 128, (128, 256), (24, 64), 64)
        self.inception4d = Inception(512, 112, (144, 288), (32, 64), 64)
        self.inception4e = Inception(528, 256, (160, 320), (32, 128), 128)

        self.inception5a = Inception(832, 256, (160, 320), (32, 128), 128)
        self.inception5b = Inception(832, 384, (192, 384), (48, 128), 128)


        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)



        self.fc = nn.Linear(1024, num_classes)

        self._initialize_weights() 


    def forward(self, x):

        x = self.features(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 正态分布初始化（适用于 ReLU）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  #B设置为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) #平均是0，方差是0.01  高斯分布，kaiming_normal是何凯明分布(He分布)
                nn.init.constant_(m.bias, 0)

    


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(device)

    print(torch.backends.cudnn.version())

    model = GoogLeNet(num_classes=10,in_channels=1).to(device)

    info = summary(model,input_size=(1,1,224,224),device=str(device))

    print(info)


