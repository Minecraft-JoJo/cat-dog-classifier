import os
import shutil
import random

# 设置随机种子，保证可复现
random.seed(42)

# 原始训练集路径
train_root = './data/training_set'   # 内部包含 cats/ 和 dogs/
test_root = './data/test_set'     # 新建的测试集目录

# 创建测试集目录
os.makedirs(os.path.join(test_root, 'cats'), exist_ok=True)
os.makedirs(os.path.join(test_root, 'dogs'), exist_ok=True)

# 每个类别要抽取的数量
samples_per_class = 500

for class_name in ['cats', 'dogs']:
    src_dir = os.path.join(train_root, class_name)
    dst_dir = os.path.join(test_root, class_name)

    # 获取该类别所有图片文件名
    all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # 随机抽取 500 张
    selected = random.sample(all_files, samples_per_class)

    for fname in selected:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        shutil.move(src_path, dst_path)  # 移动（而不是复制）
        print(f"Moved {src_path} -> {dst_path}")

print("测试集分离完成！")