import os
from torchvision import datasets
from PIL import Image

def prepare_cifar10_folders(data_root):
    save_path = os.path.join(data_root, 'CIFAR10_Binary')
    if os.path.exists(save_path):
        print(f"✅ 文件夹已存在: {save_path}")
        return

    print("⏳ 正在下载并转换 CIFAR-10，请稍候...")

    train_set = datasets.CIFAR10(root=data_root, train=True, download=True)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True)
    

    pos_indices = [0, 1, 8, 9] 
    
    for split_name, dataset in [('train', train_set), ('test', test_set)]:
        for i, (img, target) in enumerate(dataset):
            cls_folder = 'Positive' if target in pos_indices else 'Negative'
            target_dir = os.path.join(save_path, split_name, cls_folder)
            os.makedirs(target_dir, exist_ok=True)
            img.save(os.path.join(target_dir, f"{i}.png"))
            
    print(f"🎉 转换完成！保存在: {save_path}")


if __name__ == "__main__":

    my_data_root = "E:/SMM工作/0318-SMM/data" 
    

    prepare_cifar10_folders(my_data_root)