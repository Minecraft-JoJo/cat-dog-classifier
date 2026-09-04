from model import GoogLeNet
class setting:
    def __init__(self):
        self.model = GoogLeNet(num_classes=2,
                    in_channels=3
                    )

    '''
    num_classes = 10       #分类数
    in_channels = 1        #输入通道数

    '''  

 

    # --- 数据参数 ---
    data_root = './data/training_set'       # 数据存放根目录
    val_root = './data/val_set'
    test_root = './data/test_set'
    single_root = './data/single/3.png'
    batch_size = 50
    num_workers = 8            # 数据加载的并行进程数
    split = 0.8                #训练和验证集中,训练集的占有量
    input_size=224           #默认图片大小      googlenet 224,VGG 227,
    class_names = ['cat', 'dog']

    # --- 训练参数 ---
    epochs = 50                # 总训练轮数
    learning_rate = 0.0001
    weight_decay = 5e-4        # L2 正则化系数
    save_path = './best_model.pth'


    

  