import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR

from Dataset import adni2
from model_2 import HASTAN, HASTANConfig, load_roi_mapping, load_mni_coordinates


def print_log(string):
    localtime = time.asctime(time.localtime(time.time()))
    print(f"[ {localtime} ] {string}")

def cacu_metric(output, y):
    predict = torch.argmax(output, dim=-1)
    ACC = torch.sum(predict == y)
    y = y.cpu().numpy()
    predict = predict.cpu().numpy()

    # 简单计算
    ACC = metrics.accuracy_score(y, predict)
    return ACC

def main(args):
    # 1. 设备配置
    device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    if args['seed']:
        seed = args['seed']
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print_log(f"Random Seed Fixed: {seed}")

    pkl_path = args.get('pkl_path', './AAL90_Yeo7_Index0to7.pkl')
    mni_path = args.get('mni_path', './aal90_mni.npy')

    print_log(f"Loading Map: {pkl_path}")
    print_log(f"Loading MNI: {mni_path}")

    roi_mapping = load_roi_mapping(pkl_path).to(device)
    mni_coords = load_mni_coordinates(mni_path).to(device)

    hastan_config = HASTANConfig()
    for k, v in args.items():
        setattr(hastan_config, k, v)

    hastan_config.num_windows = (hastan_config.time_points - hastan_config.window_size) // hastan_config.stride + 1

    print_log(f"Model Config Loaded. Num Windows: {hastan_config.num_windows}")

    all_best_ACC = np.zeros(10)
    all_best_AUC = np.zeros(10)

    for k in args['k_folds']:
        print_log(f'===================== Fold {k} =====================')

        # 加载数据
        train_data = adni2(data=args['prefix'], split=k, mode='train')
        valid_data = adni2(data=args['prefix'], split=k, mode='test')

        # DataLoader
        workers = 2
        train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True, num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))      # <--- 必须为 True，加速 CPU 到 GPU 传输)
        valid_loader = DataLoader(valid_data, batch_size=args['batch_size'], shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))

        num_class = valid_data.get_num_class()
        num = len(train_data)

        # 实例化模型
        My_mode = HASTAN(config=hastan_config, roi_mapping=roi_mapping, mni_coords=mni_coords,
                         num_classes=num_class).to(device)

        # 优化器
        optimizer = torch.optim.AdamW(My_mode.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        # optimizer = torch.optim.SGD(My_mode.parameters(), lr=args['lr'], momentum=args['momentum'], nesterov=True,
        #                             weight_decay=args['lr_decay_rate'])
        # scheduler = CosineAnnealingLR(optimizer, T_max=args['end_epoch'], eta_min=args.get('lr_min', 1e-6))

        # Loss
        # loss_F = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        loss_F = nn.CrossEntropyLoss().to(device)#修改
        Best_ACC = 0
        Best_AUC = 0

        # Training Loop
        for epoch in range(args['start_epoch'], args['end_epoch']):
            My_mode.train()
            total_loss = 0
            correct = 0

            for i, (x, target) in enumerate(train_loader):
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)

                # Mixup
                use_mixup = args.get('use_mixup', True)  # 默认开启 Mixup
                use_mixup = False
                if use_mixup:
                    lam = np.random.beta(1.0, 1.0)
                    index = torch.randperm(x.size(0)).to(device)
                    mixed_x = lam * x + (1 - lam) * x[index]
                    y_a, y_b = target, target[index]

                    logits = My_mode(mixed_x)
                    loss = lam * loss_F(logits, y_a) + (1 - lam) * loss_F(logits, y_b)
                else:
                    logits = My_mode(x)
                    loss = loss_F(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (torch.argmax(logits, dim=1) == target).sum().item()

            # scheduler.step()

            # Validation
            My_mode.eval()
            weight = torch.tensor([1.0, 2.0]).to(device)
            val_targets = []
            val_preds = []
            val_probs = []

            with torch.no_grad():
                for x, target in valid_loader:
                    x, target = x.to(device), target.to(device)
                    logits = My_mode(x)

                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = torch.argmax(logits, dim=1)

                    val_targets.extend(target.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())

            # Metrics
            val_targets = np.array(val_targets)
            val_preds = np.array(val_preds)
            val_probs = np.array(val_probs)

            curr_acc = metrics.accuracy_score(val_targets, val_preds)
            try:
                curr_auc = metrics.roc_auc_score(val_targets, val_probs)
            except:
                curr_auc = 0.5

            if curr_acc > Best_ACC:
                Best_ACC = curr_acc
                if not os.path.exists(args['work_dir']): os.makedirs(args['work_dir'])
                torch.save(My_mode.state_dict(), os.path.join(args['work_dir'], f'best_model_fold{k}.pth'))


        all_best_ACC[k - 1] = Best_ACC
        all_best_AUC[k - 1] = Best_AUC

    print_log("\n================ Final Summary ================")
    print_log(f"Mean ACC: {np.mean(all_best_ACC )  / num:.4f}")





