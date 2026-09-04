import torch
from PIL import Image
from config import setting
from torchvision import transforms
import torch.nn.functional as F


def load_pic(cfg):
    try:
        img = Image.open(cfg.single_root).convert('RGB')  # 确保 RGB 三通道
    except Exception as e:
        print(f"无法读取图片: {e}")
        return img
    
    transform =  transforms.Compose([
            transforms.Resize((cfg.input_size,cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                             std=[0.229, 0.224, 0.225])])
    
    img = transform(img).unsqueeze(0)  # 增加 batch 维度，形状: (1, 3, 224, 224) 

    return img

def model_test(model,x):


    model = model.to(device)

    x = x.to(device)


    model.eval()


    with torch.no_grad():
        output = model(x)
        prob = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(prob, 1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()

    return confidence,predicted_idx




if  __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    print(device)

    cfg = setting()

    cfg.model.load_state_dict(torch.load('./best_model.pth'))

    img = load_pic(cfg)

    if img is None:
        print("❌ 图片加载失败，程序退出。")
        exit()

    class_names = cfg.class_names

    pros,idx = model_test(cfg.model,img)

    print(f"预测的结果是{class_names[idx]},概率是{pros:.4f}")






