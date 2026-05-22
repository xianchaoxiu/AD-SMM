"""
训练脚本
========
用法:
    python train.py --dataset brain_tumor --img_size 64 --K 6 --epochs 50 --batch_size 32 --lr 1e-3
    python train.py --dataset busi --img_size 64 --K 6 --epochs 80 --batch_size 16 --lr 5e-4
    python train.py --dataset concrete --img_size 64 --K 6 --epochs 30 --batch_size 64 --lr 1e-3 --max_samples 8000
    python train.py --dataset malaria --img_size 64 --K 6 --epochs 30 --batch_size 64 --lr 1e-3 --max_samples 8000

参数说明:
    --dataset: 数据集名称 (brain_tumor / busi / concrete / malaria)
    --data_root: 数据根目录, 包含4个数据集文件夹
    --img_size: 图像resize尺寸 (正方形)
    --K: 展开迭代层数
    --epochs: 训练轮数
    --batch_size: 批大小
    --lr: 学习率
    --max_samples: 最大样本数 (调试用, 默认None=全部)
    --use_rdb: 是否使用RDB近端算子 (默认True)
    --att_mode: 注意力加在哪一项 (fnorm / nuclear / hinge), 默认fnorm
    --save_dir: 模型保存目录
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ag_smm import AGSMM_Net, AGSMM_Net_Nuclear, AGSMM_Net_Hinge, AGSVM_Net
from utils.dataset import get_dataloader
from utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='AG-SMM Training')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['brain_tumor', 'busi', 'concrete', 'malaria', 'hymenoptera','cifar10','mushroom','kvasir', 'kthtips'])
    parser.add_argument('--data_root', type=str, default='../../data')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--K', type=int, default=6, help='展开层数')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_rdb', action='store_true', default=True)
    parser.add_argument('--no_rdb', action='store_true', default=False)
    parser.add_argument('--att_mode', type=str, default='fnorm',
                        choices=['fnorm', 'nuclear', 'hinge', 'svm'],
                        help='注意力加在哪一项: fnorm(F范数), nuclear(核范数), hinge(hinge loss)')
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, train_loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for X, y in train_loader:
        X = X.to(device)  # (B, 1, H, W)
        y = y.float().to(device)  # (B,) values in {-1, +1}

        decision = model(X)  # (B,)

        # 将标签从{-1,+1}转为{0,1}用于BCE
        y_01 = (y + 1) / 2.0
        loss = criterion(decision, y_01)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        # 预测: decision > 0 -> +1, <= 0 -> -1
        pred = torch.where(decision > 0, torch.ones_like(y), -torch.ones_like(y))
        total_correct += (pred == y).sum().item()
        total_samples += X.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_y_score = []
    total_loss = 0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.float().to(device)

        decision = model(X)
        y_01 = (y + 1) / 2.0
        loss = criterion(decision, y_01)

        pred = torch.where(decision > 0, torch.ones_like(y), -torch.ones_like(y))

        all_y_true.extend(y.cpu().numpy().tolist())
        all_y_pred.extend(pred.cpu().numpy().tolist())
        all_y_score.extend(torch.sigmoid(decision).cpu().numpy().tolist())
        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    metrics = compute_metrics(all_y_true, all_y_pred, all_y_score)
    metrics['loss'] = total_loss / total_samples
    return metrics


def main():
    args = parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 数据
    img_size = (args.img_size, args.img_size)
    train_loader, val_loader, test_loader = get_dataloader(
        args.dataset, args.data_root, img_size=img_size,
        batch_size=args.batch_size, max_samples=args.max_samples, seed=args.seed
    )

    # 模型
    use_rdb = args.use_rdb and (not args.no_rdb)
    if args.att_mode == 'fnorm':
        model = AGSMM_Net(img_size=img_size, in_channels=1, K=args.K, use_rdb=use_rdb)
    elif args.att_mode == 'nuclear':
        model = AGSMM_Net_Nuclear(img_size=img_size, in_channels=1, K=args.K, use_rdb=use_rdb)
    elif args.att_mode == 'hinge':
        model = AGSMM_Net_Hinge(img_size=img_size, in_channels=1, K=args.K, use_rdb=use_rdb)
    elif args.att_mode == 'svm':
        model = AGSVM_Net(img_size=img_size, in_channels=1, K=args.K, use_rdb=use_rdb)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: AG-SMM ({args.att_mode}), K={args.K}, use_rdb={use_rdb}, params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 计算类别权重处理不平衡
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy().tolist())
    n_pos = sum(1 for l in all_labels if l == 1)
    n_neg = sum(1 for l in all_labels if l == -1)
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    else:
        pos_weight = torch.tensor([1.0]).to(device)
    print(f"Class balance: +1={n_pos}, -1={n_neg}, pos_weight={pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 保存路径
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.att_mode}")
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0
    best_epoch = 0
    patience = 15
    no_improve = 0
    # ✅ 正确：创建日志文件
    log_file = open(os.path.join(save_dir, 'epoch_log.csv'), 'w', encoding='utf-8')
    log_file.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1,val_auc,lr\n")

    print(f"\n{'='*60}")
    print(f"Training AG-SMM ({args.att_mode}) on {args.dataset}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
              f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics.get('auc', 0):.4f} | "
              f"LR: {lr_now:.2e} | {elapsed:.1f}s")

        # ===================== 修正点1 =====================
        # ✅ 正确：放在循环内，每一轮都写入！
        log_file.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_metrics['loss']:.4f},{val_metrics['accuracy']:.4f},{val_metrics['f1']:.4f},{val_metrics.get('auc',0):.4f},{lr_now:.2e}\n")
        log_file.flush()
        # ====================================================

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
            }, os.path.join(save_dir, 'best_model.pth'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 加载最优模型测试
    print(f"\nBest val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, criterion)
    print(f"\n{'='*60}")
    print(f"Test Results ({args.dataset}, {args.att_mode}):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics.get('auc', 0):.4f}")
    print(f"{'='*60}")

    # 保存测试结果
    import json
    result_path = os.path.join(save_dir, 'test_results.json')
    with open(result_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Results saved to {result_path}")

    # ===================== 修正点2 =====================
    # ✅ 正确：关闭文件（放在main函数内）
    log_file.close()
    # ====================================================


if __name__ == '__main__':
    main()