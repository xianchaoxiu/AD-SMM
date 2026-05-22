
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class BinaryImageDataset(Dataset):

    def __init__(self, image_paths, labels, img_size=(64, 64), augment=False):
        self.image_paths = image_paths
        self.labels = labels  # +1 or -1
        self.img_size = img_size
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)  # (1, H, W), [0, 1]
        label = self.labels[idx]
        return img, label


def collect_brain_tumor(data_root):
    """Brain Tumor MRI: tumor vs notumor"""
    paths, labels = [], []
    for split in ['Training', 'Testing']:
        split_dir = os.path.join(data_root, 'Brain_Tumor_MRI_Dataset', split)
        if not os.path.isdir(split_dir):
            continue
        for cls_name in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = -1 if cls_name == 'notumor' else 1  # tumor=+1, notumor=-1
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(label)
    return paths, labels


def collect_busi(data_root):
    """BUSI: malignant(+1) vs benign(-1), 排除normal和mask图"""
    paths, labels = [], []
    busi_dir = os.path.join(data_root, 'Dataset_BUSI_with_GT')
    for cls_name in ['benign', 'malignant']:
        cls_dir = os.path.join(busi_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        label = 1 if cls_name == 'malignant' else -1
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 排除mask图 (文件名含 _mask)
                if '_mask' in fname.lower():
                    continue
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return paths, labels


def collect_concrete(data_root):
    """Concrete Crack: Positive(+1) vs Negative(-1)"""
    paths, labels = [], []
    base_dir = os.path.join(data_root, 'Concrete_Crack_Images')
    for cls_name in ['Positive', 'Negative']:
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        label = 1 if cls_name == 'Positive' else -1
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return paths, labels


def collect_malaria(data_root):
    """Malaria Cell: Parasitized(+1) vs Uninfected(-1)"""
    paths, labels = [], []
    base_dir = os.path.join(data_root, 'cell_images')
    for cls_name in ['Parasitized', 'Uninfected']:
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        label = 1 if cls_name == 'Parasitized' else -1
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return paths, labels

####新加数据集
def collect_hymenoptera(data_root):
    paths, labels = [], []
    base_dir = os.path.join(data_root, 'hymenoptera_data')

    # 例：base_dir/Positive、base_dir/Negative
    for cls_name in ['ants', 'bees']:
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        label = 1 if cls_name == 'ants' else -1
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return paths, labels

def collect_cifar10(data_root):
    paths, labels = [], []
    base_dir = os.path.join(data_root, 'cifar10')
    # 例：base_dir/Positive、base_dir/Negative
    for cls_name in ['Negative', 'Positive']:
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        label = 1 if cls_name == 'Negative' else -1
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return paths, labels

def collect_kvasir(data_root):
    paths, labels = [], []
    base_dir = os.path.join(data_root, 'Kvasir')
    # 例：base_dir/Positive、base_dir/Negative
    for cls_name in ['normal-cecum', 'polyps']:
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        label = 1 if cls_name == 'normal-cecum' else -1
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return paths, labels

def collect_kthtips(data_root):
    paths, labels = [], []
    base_dir = os.path.join(data_root, 'KTH-TIPS')
    # 例：base_dir/Positive、base_dir/Negative
    for cls_name in ['sandpaper', 'sponge']:
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        label = 1 if cls_name == 'sandpaper' else -1
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return paths, labels

DATASET_COLLECTORS = {
    'brain_tumor': collect_brain_tumor,
    'busi': collect_busi,
    'concrete': collect_concrete,
    'malaria': collect_malaria,
    'hymenoptera': collect_hymenoptera,
    'cifar10': collect_cifar10,
    'kvasir': collect_kvasir,
    'kthtips': collect_kthtips,
}


def get_dataloader(dataset_name, data_root, img_size=(64, 64), batch_size=32,
                   train_ratio=0.7, val_ratio=0.15, max_samples=None, seed=42):
    
    collector = DATASET_COLLECTORS[dataset_name]
    paths, labels = collector(data_root)

    # 打乱
    combined = list(zip(paths, labels))
    random.seed(seed)
    random.shuffle(combined)
    paths, labels = zip(*combined)
    paths, labels = list(paths), list(labels)

    if max_samples is not None:
        paths = paths[:max_samples]
        labels = labels[:max_samples]

    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_paths, train_labels = paths[:n_train], labels[:n_train]
    val_paths, val_labels = paths[n_train:n_train + n_val], labels[n_train:n_train + n_val]
    test_paths, test_labels = paths[n_train + n_val:], labels[n_train + n_val:]

    train_ds = BinaryImageDataset(train_paths, train_labels, img_size, augment=True)
    val_ds = BinaryImageDataset(val_paths, val_labels, img_size, augment=False)
    test_ds = BinaryImageDataset(test_paths, test_labels, img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[{dataset_name}] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    pos_train = sum(1 for l in train_labels if l == 1)
    neg_train = sum(1 for l in train_labels if l == -1)
    print(f"  Train class distribution: +1={pos_train}, -1={neg_train}")

    return train_loader, val_loader, test_loader
