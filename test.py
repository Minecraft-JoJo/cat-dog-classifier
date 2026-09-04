import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets import ima
from torchvision import transforms
import torch.utils.data as Data
from config import setting
import time


def data_process(cfg):
    transforming_val =  transforms.Compose([
            transforms.Resize((cfg.input_size,cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                             std=[0.229, 0.224, 0.225])])

    data_test = ImageFolder(root=cfg.test_root,transform=transforming_val)

   
    x_test= Data.DataLoader(dataset=data_test,batch_size=cfg.batch_size,num_workers=cfg.num_workers)
    return x_test


def model_test(model,x):

    num = 0

    each_test_acc_total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    model.eval()

    print("开始推理验证")

    start_time = time.time()


    with torch.no_grad():


        for step, (b_x,b_y) in enumerate(x):

            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
                
            pre_lab = torch.argmax(output,dim=1)  
        
            num += b_x.size(0) 
                
            each_test_acc_total += torch.sum(pre_lab == b_y)  


    acc = (each_test_acc_total/num).double().item()

    end_time = time.time()

    time_total = end_time - start_time

    minutes, seconds = divmod(time_total, 60)

    print(f"推理共耗时: {int(minutes)} 分 {seconds:.1f} 秒")

    return acc


if __name__ == '__main__':

    cfg = setting()

    cfg.model.load_state_dict(torch.load('./best_model.pth'))

    x = data_process(cfg)

    acc = model_test(cfg.model,x)

    print(f"准确率是{acc:.4f}")
