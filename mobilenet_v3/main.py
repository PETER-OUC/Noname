#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNet v3 统一入口脚本
支持训练 (--mode train) 和预测 (--mode predict) 两种模式
移除了 d2l 库的依赖
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn.metrics as sm
import torch
import torch.optim as optim
import torch.quantization
from scipy import datasets, interpolate, ndimage, signal
from scipy.io import loadmat
from scipy.io.wavfile import read as wav_read
from scipy.ndimage import zoom
from scipy.signal import stft
from scipy.signal.windows import hamming
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

# ======================
# 替代 d2l 的辅助函数/类
# ======================


def try_gpu():
    """如果GPU可用则返回cuda设备，否则返回cpu"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Accumulator:
    """在 n 个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """计算分类准确率"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if device is None and isinstance(net, nn.Module):
        device = next(iter(net.parameters())).device
    net.eval()
    metric = Accumulator(2)  # 正确预测数，预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# ======================
# 网络结构定义
# ======================

class ResidualDSBlock(nn.Module):
    """带残差连接的 Depthwise Separable Block"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=False),
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_residual = (stride == 1 and in_ch == out_ch)
        self.relu6 = nn.ReLU6(inplace=False)
    
    def forward(self, x):
        if self.use_residual:
            return self.relu6(self.conv(x) + x)
        else:
            return self.relu6(self.conv(x))


class MobileNet(nn.Module):
    """MobileNet with aggressive early downsampling"""
    def __init__(self, config):
        super().__init__()
        layers = []
        
        stem = config["stem"]
        layers.append(nn.Conv2d(
            config["input_channels"], stem["out_channels"],
            kernel_size=stem["kernel_size"], stride=stem["stride"],
            padding=stem["padding"], bias=False
        ))
        layers.append(nn.BatchNorm2d(stem["out_channels"]))
        layers.append(nn.ReLU6(inplace=False))
        
        if config.get("stem_pool", None):
            pool_cfg = config["stem_pool"]
            layers.append(nn.MaxPool2d(
                kernel_size=pool_cfg["kernel_size"],
                stride=pool_cfg["stride"],
                padding=pool_cfg.get("padding", 0)
            ))
        
        for block in config["blocks"]:
            layers.append(ResidualDSBlock(
                block["in_channels"], block["out_channels"], stride=block["stride"]
            ))
        
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.fc = nn.Linear(
            config["classifier"]["in_features"], config["num_classes"]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_mobilenet_from_config(config):
    return MobileNet(config)


# ======================
# 数据集定义
# ======================

class MyDataset(Dataset):
    def __init__(self, data_in, label_1, transform=None):
        self.X = data_in
        self.y = label_1
        self.transforms = transform
        
    def __getitem__(self, index):
        ldct = self.transforms(self.X[index, :, :, :]) if self.transforms else self.X[index]
        hdct = self.y[index]
        return ldct, hdct
        
    def __len__(self):
        return self.X.shape[0]


def load_data_my(train_data, test_data, batch_size):
    """加载训练和测试数据"""
    return (data.DataLoader(train_data, batch_size, shuffle=True, num_workers=0),
            data.DataLoader(test_data, batch_size, shuffle=False, num_workers=0))


# ======================
# 默认网络配置
# ======================

DEFAULT_NETWORK_CONFIG = {
    "input_channels": 1,
    "num_classes": 4,
    "dropout_rate": 0.2,
    "stem": {
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1
    },
    "blocks": [
        {"in_channels": 16, "out_channels": 32, "stride": 2},
        {"in_channels": 32, "out_channels": 32, "stride": 1},
        {"in_channels": 32, "out_channels": 64, "stride": 2},
    ],
    "classifier": {
        "in_features": 64
    }
}


# ======================
# 训练模式
# ======================

def list_resume_checkpoints():
    """扫描当前目录下可恢复的训练断点"""
    resume_dirs = []
    for d in sorted(os.listdir(os.getcwd()), reverse=True):
        d_path = os.path.join(os.getcwd(), d)
        if not os.path.isdir(d_path):
            continue
        latest = os.path.join(d_path, "latest_checkpoint.pth")
        if os.path.exists(latest):
            resume_dirs.append(d_path)
    return resume_dirs


def prompt_resume_or_new():
    """询问用户恢复训练还是开始新训练"""
    resume_dirs = list_resume_checkpoints()
    if not resume_dirs:
        return None, False
    print("\n" + "="*60)
    print("检测到以下可恢复的训练断点:")
    print("-"*60)
    for i, d in enumerate(resume_dirs):
        try:
            ckpt = torch.load(os.path.join(d, "latest_checkpoint.pth"), map_location='cpu', weights_only=False)
            epoch = ckpt.get('epoch', -1)
            best_acc = ckpt.get('best_acc', 0)
            print(f"  [{i+1}] {os.path.basename(d)}")
            print(f"        已完成 epoch: {epoch+1} | 最佳精度: {best_acc:.4f}")
        except Exception as e:
            print(f"  [{i+1}] {os.path.basename(d)} (读取失败: {e})")
    print("-"*60)
    print("  [0] 开始新的训练")
    print("="*60)
    while True:
        try:
            choice = input("\n请选择 [0-{}]: ".format(len(resume_dirs))).strip()
            if choice == "0":
                return None, False
            idx = int(choice) - 1
            if 0 <= idx < len(resume_dirs):
                return resume_dirs[idx], True
        except (ValueError, IndexError):
            pass
        print("无效选择，请重新输入。")


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train():
    """训练主函数"""
    import matplotlib.pyplot as plt
    from IPython import get_ipython

    torch.manual_seed(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 设备
    device = try_gpu()

    # 尝试获取 notebook 路径
    try:
        ip = get_ipython()
        if ip and '__vsc_ipynb_file__' in ip.user_ns:
            path = ip.user_ns['__vsc_ipynb_file__']
            print(f"Notebook path: {path}")
        else:
            path = None
    except Exception:
        path = None

    # 图像变换
    transform = transforms.Compose([transforms.ToTensor()])

    # ----------------------
    # 数据加载
    # ----------------------
    dir_path = os.getcwd()
    npy_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.npy')]

    print(f"Found {len(npy_files)} .npy files")
    for f in npy_files:
        print(f"  - {f}")

    # 加载数据文件
    X_train = X_test = X_val = None
    y_train = y_test = y_val = None
    for npy_file in npy_files:
        prefix = npy_file[:4]
        arr = np.load(npy_file)
        if prefix == 'X_tr':
            X_train = arr
        elif prefix == 'X_te':
            X_test = arr
        elif prefix == 'X_va':
            X_val = arr
        elif prefix == 'y_tr':
            y_train = arr
        elif prefix == 'y_te':
            y_test = arr
        elif prefix == 'y_va':
            y_val = arr

    if X_train is None or y_train is None:
        print("错误：未找到训练数据文件（X_tr*.npy / y_tr*.npy）")
        return

    # 保存原始标签用于后续验证
    y_1 = y_val.copy() if y_val is not None else None

    # 转换为tensor
    y_train = torch.tensor(y_train, dtype=torch.uint8)
    if y_test is not None:
        y_test = torch.tensor(y_test, dtype=torch.uint8)
    if y_val is not None:
        y_val = torch.tensor(y_val, dtype=torch.uint8)

    # 创建数据集
    train_data = MyDataset(X_train, y_train, transform)
    test_data = MyDataset(X_test, y_test, transform) if X_test is not None else None
    val_data = MyDataset(X_val, y_val, transform) if X_val is not None else None

    # 创建验证集数据加载器
    val_iter = data.DataLoader(val_data, 32, shuffle=False, num_workers=0) if val_data is not None else None

    print(f"\nDataset shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    if X_test is not None:
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    if X_val is not None:
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")

    # 获取输入形状（单通道）
    input_1 = X_train[0, :, :, 0]
    input_shape = input_1[np.newaxis, :].shape
    print(f"\nInput shape (single channel): {input_shape}")

    # ----------------------
    # 输出目录设置
    # ----------------------
    resume_dir, is_resume = prompt_resume_or_new()

    if is_resume:
        print(f"\n恢复训练，目录: {resume_dir}")
        checkpoint = torch.load(os.path.join(resume_dir, "latest_checkpoint.pth"), map_location='cpu', weights_only=False)
        output_dir = resume_dir
        File_name = checkpoint['File_name']
        timestamp = checkpoint['timestamp']
        NETWORK_CONFIG = checkpoint['NETWORK_CONFIG']
        print(f"  已恢复配置: {File_name}")
        print(f"  输出目录: {output_dir}")
    else:
        if path:
            File_name = path[len(os.getcwd())+1:-6] + '_' + dir_path[-3:]
        else:
            File_name = "mobile_net_" + dir_path[-3:]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), f"{File_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        NETWORK_CONFIG = copy.deepcopy(DEFAULT_NETWORK_CONFIG)

    # 复制数据预处理文件（如果存在）
    dataprocess_source = "dataprocess_unified.ipynb"
    if os.path.exists(dataprocess_source):
        dataprocess_dest = os.path.join(output_dir, "dataprocess_unified.ipynb")
        shutil.copy2(dataprocess_source, dataprocess_dest)
        print(f"Copied {dataprocess_source} to output directory")
    else:
        print(f"Warning: {dataprocess_source} not found in current directory")

    # 保存训练信息
    training_info_path = os.path.join(output_dir, "training_info.txt")
    with open(training_info_path, "w", encoding="utf-8") as f:
        f.write(f"Training Start Time: {timestamp}\n")
        f.write(f"File Name: {File_name}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Current Working Directory: {os.getcwd()}\n")
    print(f"Training info saved to: {training_info_path}")

    # ----------------------
    # 构建网络
    # ----------------------
    net = build_mobilenet_from_config(NETWORK_CONFIG)

    if is_resume:
        net.load_state_dict(checkpoint['model_state_dict'])
        print("  -> 已恢复模型权重")

    print("MobileNet Fast-Downsampling model defined successfully.")
    print(f"Network config: {NETWORK_CONFIG}")

    # ----------------------
    # 模型摘要和参数量统计
    # ----------------------
    try:
        from torchinfo import summary
        print("\n=== MobileNet 架构详情 ===")
        summary(net, input_size=(1, 1, 100, 100), device='cpu')
    except ImportError:
        print("Warning: torchinfo not installed. Use: pip install torchinfo")
        total_params = sum(p.numel() for p in net.parameters())
        print(f"\nTotal parameters: {total_params:,}")

    total_params = sum(p.numel() for p in net.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"FP32模型大小: {total_params*4/1024:.2f} KB")
    print(f"INT8量化后预计: {total_params/1024:.2f} KB (远小于50KB限制)")

    # ----------------------
    # 训练配置
    # ----------------------
    lr, num_epochs, batch_size = 0.005, 100, 2048

    if test_data is None:
        print("错误：未找到测试数据，无法训练")
        return

    train_iter, test_iter = load_data_my(train_data, test_data, batch_size)

    # 记录初始化
    if is_resume:
        ckpt_num_epochs = checkpoint['num_epochs']
        ckpt_record = checkpoint['record_list']
        if num_epochs > ckpt_record.shape[0]:
            record_list = np.zeros((num_epochs, 3))
            record_list[:ckpt_record.shape[0]] = ckpt_record
        else:
            num_epochs = ckpt_num_epochs
            record_list = ckpt_record
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch'] + 1
        print(f"  -> 恢复训练状态: 从 epoch {start_epoch} 开始, best_acc={best_acc:.4f}")
    else:
        record_list = np.zeros((num_epochs, 3))
        best_acc = 0.0
        start_epoch = 0

    # 权重初始化（恢复训练时不应重新初始化！）
    if not is_resume:
        net.apply(init_weights)

    print('Training on', device)
    net.to(device)

    # torch.compile 优化
    compile_enabled = False
    try:
        compiled_net = torch.compile(net, backend="aot_eager", fullgraph=False)
        compile_enabled = True
        print('torch.compile enabled (backend=aot_eager)')
    except Exception as e:
        compiled_net = net
        print(f'torch.compile failed: {e}')
        print('Training in eager mode.')

    # 自动混合精度
    try:
        scaler = torch.amp.GradScaler('cuda')
    except TypeError:
        scaler = torch.cuda.amp.GradScaler()

    if is_resume:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("  -> 已恢复AMP scaler状态")

    print('AMP enabled')

    # 优化器
    optimizer = torch.optim.SGD(compiled_net.parameters(), lr=lr, momentum=0.9)

    if is_resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  -> 已恢复优化器状态")

    loss = nn.CrossEntropyLoss()

    # 保存网络配置
    network_config_path = os.path.join(output_dir, "network_config.json")
    with open(network_config_path, "w", encoding="utf-8") as pf:
        json.dump(NETWORK_CONFIG, pf, indent=4, ensure_ascii=False)
    print(f"Network config saved to: {network_config_path}")

    num_batches = len(train_iter)

    print(f"\nTraining configuration:")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {num_batches}")
    print(f"\n开始训练...")

    # ----------------------
    # 训练循环
    # ----------------------
    T1 = time.time()

    training_interrupted = False
    last_completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, num_epochs):
            metric = Accumulator(3)
            compiled_net.train()
            
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                
                # AMP前向传播
                with torch.amp.autocast('cuda'):
                    y_hat = compiled_net(X)
                    l = loss(y_hat, y)
                
                scaler.scale(l).backward()
                scaler.step(optimizer)
                scaler.update()
                
                with torch.no_grad():
                    metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
                
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                
                if (i + 1) % num_batches == 0:
                    elapsed = time.time() - T1
                    print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{num_batches}] "
                          f"loss={train_l:.4f}, train_acc={train_acc:.4f}, time={elapsed:.1f}s")
            
            # 测试集评估
            test_acc = evaluate_accuracy_gpu(compiled_net, test_iter, device)
            
            record_list[epoch, 0] = train_l
            record_list[epoch, 1] = train_acc
            record_list[epoch, 2] = test_acc
            
            last_completed_epoch = epoch
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net, os.path.join(output_dir, File_name + '.pth'))
                print(f"  -> New best model saved! (test_acc: {test_acc:.4f})")
            else:
                print(f"  -> test_acc: {test_acc:.4f} (best: {best_acc:.4f})")
            
            if (epoch + 1) % 100 == 0:
                checkpoint_path = os.path.join(output_dir, File_name + f'_epoch{epoch+1}.pth')
                torch.save(net, checkpoint_path)
                print(f"  -> Checkpoint saved at epoch {epoch+1}")

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("检测到训练中断 (Ctrl+C)")
        print(f"{'='*60}")
        print("正在保存断点...")
        
        checkpoint = {
            'epoch': last_completed_epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc,
            'record_list': record_list,
            'output_dir': output_dir,
            'File_name': File_name,
            'timestamp': timestamp,
            'num_epochs': num_epochs,
            'lr': lr,
            'batch_size': batch_size,
            'NETWORK_CONFIG': NETWORK_CONFIG,
        }
        ckpt_path = os.path.join(output_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, ckpt_path)
        print(f"  -> 断点已保存: {ckpt_path}")
        print(f"  -> 已完成 epoch: {last_completed_epoch+1} | 下次将从 epoch: {last_completed_epoch+2} 继续")
        print(f"{'='*60}")
        training_interrupted = True

    T2 = time.time()
    print(f'\n程序运行时间: {(T2 - T1):.2f} 秒')

    if training_interrupted:
        print("\n训练已中断，已保存断点。下次运行可选择恢复训练。")
        print(f"断点目录: {output_dir}")
        sys.exit(0)

    # ----------------------
    # 保存训练记录和参数
    # ----------------------
    with open(os.path.join(output_dir, File_name + '.csv'), 'w', newline='') as f:
        f_csv = csv.writer(f, dialect='excel')
        f_csv.writerows(record_list)
    print(f"Training record saved to: {File_name}.csv")

    params = {
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "best_accuracy": best_acc,
        "training_time_seconds": T2 - T1,
        "training_start_time": timestamp,
        "training_end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_shape": list(input_shape),
        "num_classes": 4,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "val_samples": len(X_val) if X_val is not None else 0,
        "network_architecture": "MobileNet_Tiny",
        "total_parameters": total_params,
        "model_size_kb": total_params * 4 / 1024,
        "quantized_size_kb": total_params / 1024,
        "optimizer": "SGD",
        "loss_function": "CrossEntropyLoss",
        "file_name": File_name,
        "output_directory": output_dir,
        "network_config": NETWORK_CONFIG,
    }

    params_path = os.path.join(output_dir, File_name + "_params.json")
    with open(params_path, "w", encoding="utf-8") as pf:
        json.dump(params, pf, indent=4, ensure_ascii=False)
    print(f"Parameters saved to: {params_path}")

    # ----------------------
    # 导出 ONNX
    # ----------------------
    model_path = os.path.join(output_dir, File_name + '.pth')
    if os.path.exists(model_path):
        model_1 = torch.load(model_path, map_location='cpu', weights_only=False)
        model_1 = model_1.cpu()
        model_1.eval()
        
        x = torch.randn(1, *input_shape)
        export_onnx_file = os.path.join(output_dir, File_name) + ".onnx"
        
        torch.onnx.export(model_1, x, export_onnx_file,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"},
                                       "output": {0: "batch_size"}})
        print(f"ONNX model exported: {export_onnx_file}")
    else:
        print(f"Warning: Model file not found at {model_path}")
        model_1 = net.cpu()
        model_1.eval()

    # ----------------------
    # 验证集评估（FP32）
    # ----------------------
    if val_iter is not None and y_1 is not None:
        net_val = model_1.to(device)
        net_val.eval()

        y_hat_all = np.zeros((0,))
        with torch.no_grad():
            for X, y in val_iter:
                X = X.to(device)
                y_hat = net_val(X)
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    y_hat = y_hat.cpu()
                    y_hat_all = np.concatenate((y_hat_all, y_hat.argmax(dim=1).numpy()))

        acc_float = accuracy_score(y_hat_all, y_1)
        print(f'\nFP32 Validation Accuracy: {acc_float:.4f}')

        # 混淆矩阵（FP32）
        cm = sm.confusion_matrix(y_1, y_hat_all)
        im = []
        for i in range(4):
            im.append(cm[i, :] / (sum(cm[i, :]) + 1e-8))

        fig = plt.figure(figsize=(10, 10), dpi=300)
        sns.heatmap(np.array(im), annot=True, linewidths=2, cmap='binary',
                    cbar=False, annot_kws={"fontsize": 30})
        plt.xticks([0.5, 1.5, 2.5, 3.5], 
                   [r'$Nothing$', r'$Speedboat$', r'$Dolphin$', r'$Whale$'], fontsize=20)
        plt.yticks([0.5, 1.5, 2.5, 3.5], 
                   [r'$Nothing$', r'$Speedboat$', r'$Dolphin$', r'$Whale$'], fontsize=20)
        plt.xlabel("Pre label", fontsize=20)
        plt.ylabel("True label", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, File_name + '.png'), 
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Confusion matrix saved to: {File_name}.png")
    else:
        acc_float = 0.0
        print("警告：未找到验证集，跳过验证评估")

    # ----------------------
    # 自动预测：用最优模型处理当前目录下的 wav 文件
    # ----------------------
    wav_files_train = glob.glob('*.wav')
    best_model_path = os.path.join(output_dir, File_name + '.pth')
    if wav_files_train and os.path.exists(best_model_path):
        print(f"\n=== 自动预测当前目录下的 WAV 文件 ===")
        print(f"使用最优模型: {best_model_path}")
        try:
            process_single_model(
                model_path=best_model_path,
                model_name=File_name,
                epoch='best',
                wav_files=wav_files_train,
                output_base_dir=output_dir
            )
        except Exception as e:
            print(f"自动预测时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not wav_files_train:
            print("\n当前目录下未找到 .wav 文件，跳过自动预测。")
        else:
            print(f"\n未找到最优模型文件: {best_model_path}，跳过自动预测。")

    # ----------------------
    # 训练总结
    # ----------------------
    print(f"\n=== 训练完成 ===")
    print(f"最佳测试精度: {best_acc:.4f}")
    if val_iter is not None:
        print(f"FP32验证精度: {acc_float:.4f}")
    print(f"模型参数量: {total_params} ({total_params/1024:.2f}K)")
    print(f"FP32模型大小: {total_params*4/1024:.2f} KB")
    print(f"预计INT8文件大小: ~{total_params/1024:.1f}KB (含量化参数)")
    print(f"所有结果保存至: {output_dir}")
    print(f"\n提示: 如需 INT8 量化，请使用独立的量化脚本处理本目录下的 .pth 模型。")


# ======================
# 预测模式
# ======================

NORMALIZATION_TYPE = 'global'  # 可选: 'column' 或 'global'


def load_network_config(output_dir):
    """从输出目录加载网络配置文件"""
    config_path = os.path.join(output_dir, "network_config.json")
    params_path_pat = os.path.join(output_dir, "*_params.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"Loaded network config from: {config_path}")
            return config
        except Exception as e:
            print(f"Warning: Failed to load {config_path}: {e}")

    params_files = glob.glob(params_path_pat)
    if params_files:
        try:
            with open(params_files[0], "r", encoding="utf-8") as f:
                params = json.load(f)
            if "network_config" in params:
                print(f"Loaded network config from: {params_files[0]}")
                return params["network_config"]
        except Exception as e:
            print(f"Warning: Failed to load params file: {e}")

    print("Warning: Could not find network config file, using default config")
    return None


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"FP32 size: {total_params*4/1024:.2f} KB")
    print(f"INT8 size: ~{total_params/1024:.2f} KB")
    return total_params


def find_output_directories():
    current_dir = os.getcwd()
    output_dirs = []
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path):
            if item.endswith('_output') or re.search(r'_\d{8}_\d{6}$', item):
                output_dirs.append(item_path)
    return output_dirs


def select_output_directory():
    selected_dir = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        try:
            existing_root = tk._default_root
            if existing_root:
                existing_root.destroy()
                tk._default_root = None
        except Exception:
            pass
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        initial_dir = os.getcwd()
        print("\n>>> 正在弹出目录选择对话框，请选择包含模型的输出目录...")
        print("(如果对话框未显示，请检查窗口是否被最小化到任务栏)\n")
        print("提示: 如果对话框多次无法弹出，可以直接输入目录路径\n")
        
        selected_dir = filedialog.askdirectory(
            title="选择模型输出目录",
            initialdir=initial_dir,
            parent=root
        )
        
        root.attributes('-topmost', False)
        root.quit()
        root.destroy()
        try:
            tk._default_root = None
        except Exception:
            pass
            
    except Exception as e:
        print(f"\n! 对话框打开失败: {e}")
        print("! 切换到命令行输入模式...\n")
    
    if not selected_dir:
        try:
            print("\n" + "-" * 50)
            print("请手动输入模型目录路径（或直接回车使用自动发现）:")
            print("-" * 50)
            user_input = input("> ").strip()
            if user_input and os.path.isdir(user_input):
                selected_dir = user_input
                print(f"\n✓ 已使用输入的目录: {selected_dir}")
            elif user_input:
                print(f"\n! 路径不存在: {user_input}")
        except KeyboardInterrupt:
            print("\n! 用户取消输入")
        except Exception as e:
            print(f"\n! 输入失败: {e}")
    
    if selected_dir:
        return [selected_dir]
    else:
        print("\n! 将使用自动发现模式...\n")
        return None


def get_all_pth_models(output_dir):
    pth_files = []
    for file in os.listdir(output_dir):
        if file.endswith('.pth') and not file.endswith('_int8.pth'):
            file_path = os.path.join(output_dir, file)
            base_name = file[:-4]
            epoch_match = re.search(r'_epoch(\d+)$', base_name)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                model_name = base_name
            else:
                epoch = 'best'
                model_name = base_name
            pth_files.append((file_path, model_name, epoch))
    pth_files.sort(key=lambda x: (x[2] != 'best', x[2] if x[2] != 'best' else 999999))
    return pth_files


# 预测预处理参数
Sample_time_len = 2
window_length_s = 0.02
overlap_ratio = 0.0
overlap_ratio_frame = 0.5
frequency_choose_low, frequency_choose_high = 3e3, 13e3
output_channel = 1

if overlap_ratio == 0:
    time_bins = np.floor(Sample_time_len / window_length_s)
else:
    time_bins = np.floor((Sample_time_len - window_length_s) / (window_length_s * (1 - overlap_ratio))) + 2

nfft = 1024
freq_bins = np.floor((frequency_choose_high - frequency_choose_low) * (nfft/48000))
time_bins, freq_bins = int(time_bins), int(freq_bins)


def normalize_columns_to_255_vectorized(X):
    if len(X.shape) != 2:
        raise ValueError("输入必须是2D数组")
    min_vals = np.min(X, axis=0, keepdims=True)
    max_vals = np.max(X, axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_normalized = ((X - min_vals) / (range_vals + 1e-8)) * 255
    return X_normalized


def normalize_global_to_255(X):
    X_min, X_max = np.min(X), np.max(X)
    X_normalized = ((X - X_min) / (X_max - X_min + 1e-8)) * 255
    return X_normalized


def normalize_data(X):
    if NORMALIZATION_TYPE == 'column':
        return normalize_columns_to_255_vectorized(X)
    else:
        return normalize_global_to_255(X)


# 生成验证集标签
y_val_1s = np.concatenate((
    np.zeros((1,25))+2, np.zeros((1,5)),
    np.zeros((1,25))+1, np.zeros((1,5)), 
    np.zeros((1,25))+3, np.zeros((1,5)),
    np.zeros((1,25))+2, np.zeros((1,5)),
    np.zeros((1,25))+1, np.zeros((1,5)), 
    np.zeros((1,25))+3, np.zeros((1,5)),
    np.zeros((1,25))+2, np.zeros((1,5)),
    np.zeros((1,25))+1, np.zeros((1,5)), 
    np.zeros((1,25))+3, np.zeros((1,5)),
    np.zeros((1,25))+2, np.zeros((1,5)),
    np.zeros((1,25))+1, np.zeros((1,5)), 
    np.zeros((1,25))+3, np.zeros((1,5))
), axis=1).flatten()

sample_length_2s = 2
overlap_ratio_val = 0.5
step_size = sample_length_2s * (1 - overlap_ratio_val)
total_duration = len(y_val_1s)
num_new_samples = int((total_duration - sample_length_2s) / step_size) + 1

y_val_2s = []
for i in range(num_new_samples):
    start_idx = int(i * step_size)
    end_idx = start_idx + sample_length_2s
    window_labels = y_val_1s[start_idx:end_idx]
    non_zero_labels = window_labels[window_labels != 0]
    if len(non_zero_labels) > 0:
        new_label = non_zero_labels[0]
    else:
        new_label = 0
    y_val_2s.append(new_label)

y_val_2s = np.array(y_val_2s)
y_val_pred = y_val_2s


def process_single_model(model_path, model_name, epoch, wav_files, output_base_dir):
    network_config = load_network_config(output_base_dir)
    if network_config is None:
        network_config = DEFAULT_NETWORK_CONFIG
        print("Using default network config")

    net = build_mobilenet_from_config(network_config)
    print(f"Network built with config: {network_config}")

    device = try_gpu()
    model = torch.load(model_path, weights_only=False)
    model = model.to(device)
    model.eval()

    print_model_stats(model)
    
    print(f"\n{'='*60}")
    print(f"Processing Model: {model_name}")
    print(f"Epoch: {epoch}")
    print(f"{'='*60}")
    
    model_result_dir = output_base_dir
    os.makedirs(model_result_dir, exist_ok=True)
    
    results_summary = []
    transform_pred = transforms.Compose([transforms.ToTensor()])
    
    for wav_file in wav_files:
        file_name = wav_file[:-4] if wav_file.endswith('.wav') else wav_file
        wav_file_path = wav_file
        
        print(f"\nProcessing: {os.path.basename(wav_file)}")
        
        Fs, data_signal = wav_read(wav_file_path)
        if data_signal.ndim > 1:
            data_signal = data_signal[:, 0]
        
        data_signal = np.asarray(data_signal).ravel()
        data_signal_biaoting = data_signal
        
        Temp_sample_num = 359
        X_val = np.zeros((Temp_sample_num, freq_bins, time_bins, output_channel), dtype=np.uint8)
        
        sample_idex = 0
        signal_duration = len(data_signal_biaoting) / Fs
        sample_step = Sample_time_len * (1 - overlap_ratio_frame)
        frame_num = np.floor((signal_duration - Sample_time_len) / sample_step) + 1
        
        window_len = int(window_length_s * Fs)
        window_overlap = int(overlap_ratio * window_len)

        f, t, Zxx = stft(data_signal_biaoting, fs=Fs, window='hamming',
                        nperseg=window_len, noverlap=window_overlap, nfft=nfft)
        
        Frequency_start_index = np.argmin(np.abs(f - (frequency_choose_low))) 
        Frequency_end_index = Frequency_start_index + freq_bins

        stft_time_resolution = t[1] - t[0]
        samples_per_frame = int(Sample_time_len / stft_time_resolution)
        energy_list = np.zeros((1, 354))

        group_d_send = np.zeros([freq_bins,]) + 140
        freq_range_original = np.linspace(3e3, 12950, len(group_d_send))
        freq_range_target = np.linspace(frequency_choose_low, frequency_choose_high-int(1/window_length_s), freq_bins)
        f_interp = interpolate.interp1d(freq_range_original, group_d_send, kind='cubic', fill_value='extrapolate')
        group_d_interp = f_interp(freq_range_target)
        group = group_d_interp.reshape(-1, 1)
        group_linear = 10 ** ((group-np.mean(group)) / 20)
        
        for frame_one in range(int(frame_num)):
            start_time = frame_one * sample_step
            start_idx = np.argmin(np.abs(t - start_time))
            end_idx = start_idx + samples_per_frame
            
            if end_idx <= len(t):
                X_temp = np.abs(Zxx[Frequency_start_index:Frequency_end_index,
                                   start_idx:end_idx])
                X_temp_1 = X_temp
                X_temp = X_temp / group_linear
                if X_temp.shape[1] < time_bins:
                    padding = time_bins - X_temp.shape[1]
                    X_temp = np.pad(X_temp, ((0, 0), (0, padding)), mode='constant')
                
                X_temp_normalized = normalize_data(X_temp)
                X_temp_uint8 = np.floor(X_temp_normalized).astype(np.uint8)
                
                if sample_idex < Temp_sample_num:
                    energy_list[0, sample_idex] = np.sum(X_temp_1**2)
                    X_val[sample_idex, :, :, 0] = X_temp_uint8
                    sample_idex += 1
        
        val_data = MyDataset(X_val, y_val_pred, transform_pred)
        val_iter = data.DataLoader(val_data, X_val.shape[0], shuffle=False, num_workers=0)
        
        y_hat_all_0 = np.zeros((0,))
        with torch.no_grad():
            for X, y in val_iter:
                X = X.to(device)
                y_hat = model(X)
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    y_hat = y_hat.cpu()
                    y_hat_all_0 = np.concatenate((y_hat_all_0, y_hat.argmax(dim=1).numpy()))
        
        y_hat_all_2 = y_hat_all_0[:-5]
        y_val_1 = y_val_pred[:-5]

        y_hat_all_4 = y_hat_all_2.copy()
        
        # 平滑处理
        changes = []
        NN1 = len(y_hat_all_4)
        for nnnn in range(1, NN1 - 1):
            if y_hat_all_4[nnnn-1] == y_hat_all_4[nnnn+1]:
                changes.append((nnnn, y_hat_all_4[nnnn-1]))
        
        for nnnn, new_value in changes:
            y_hat_all_4[nnnn] = new_value
        
        acc = accuracy_score(y_hat_all_4, y_val_1)
        print(f"  Accuracy: {acc:.4f}")
        
        cm = sm.confusion_matrix(y_val_1, y_hat_all_4)
        im = []
        for i in range(4):
            im.append(cm[i, :] / (sum(cm[i, :]) + 1e-8))
        
        fig = plt.figure(figsize=(10, 10), dpi=300)
        sns.heatmap(np.array(im), annot=True, linewidths=2,
                   cmap='binary', cbar=False, annot_kws={"fontsize": 30})
        plt.xticks([0.5, 1.5, 2.5, 3.5],
                  [r'$Nothing$', r'$Speedboat$', r'$Dolphin$', r'$Whale$'],
                  fontsize=20)
        plt.yticks([0.5, 1.5, 2.5, 3.5],
                  [r'$Nothing$', r'$Speedboat$', r'$Dolphin$', r'$Whale$'],
                  fontsize=20)
        plt.xlabel("Pre label", fontsize=20)
        plt.ylabel("True label", fontsize=20)
        plt.tight_layout()
        
        if epoch == 'best':
            save_name = f"{file_name}_best_acc.png"
        else:
            save_name = f"{file_name}_epoch{epoch}_acc.png"
        
        save_path = os.path.join(model_result_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  Saved: {save_name}")
        
        cm_normalized = np.array(im)
        labels = ['Nothing', 'Speedboat', 'Dolphin', 'Whale']
        
        cm_data = {
            'wav_file': wav_file,
            'accuracy': acc,
            'model': model_name,
            'epoch': epoch
        }
        
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                cm_data[f'True_{true_label}_Pred_{pred_label}'] = int(cm[i, j])
        
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                cm_data[f'True_{true_label}_Pred_{pred_label}_Norm'] = float(cm_normalized[i, j])
        
        for i, label in enumerate(labels):
            total = sum(cm[i, :])
            cm_data[f'Accuracy_{label}'] = float(cm_normalized[i, i]) if total > 0 else 0.0
            cm_data[f'Total_{label}'] = int(total)
        
        results_summary.append(cm_data)
    
    if results_summary:
        cm_df = pd.DataFrame(results_summary)
        csv_filename = f"confusion_matrix_{model_name}.csv"
        cm_csv_path = os.path.join(output_base_dir, csv_filename)
        cm_df.to_csv(cm_csv_path, index=False, encoding='utf-8-sig')
        print(f"  Confusion matrix CSV saved: {csv_filename}")
    
    return results_summary


def predict():
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("Libraries imported successfully!")
    print(f'时频矩阵维度：时间轴 {time_bins}，频率轴 {freq_bins}')
    print(f'归一化方式：{"列归一化" if NORMALIZATION_TYPE == "column" else "全局归一化"}')
    
    print("=" * 70)
    print("步骤1: 选择模型输出目录")
    print("=" * 70)
    
    output_directories = select_output_directory()
    if output_directories is None:
        print("正在自动发现输出目录...")
        output_directories = find_output_directories()
    
    if not output_directories:
        print("错误：未找到任何输出目录！")
        return
    
    print(f"\n发现 {len(output_directories)} 个输出目录:")
    for i, dir_path in enumerate(output_directories, 1):
        print(f"  {i}. {os.path.basename(dir_path)}")
        pth_models = get_all_pth_models(dir_path)
        print(f"     模型数量: {len(pth_models)}")
        for model_path, model_name, epoch in pth_models[:3]:
            print(f"       - {model_name} (epoch={epoch})")
        if len(pth_models) > 3:
            print(f"       ... 还有 {len(pth_models)-3} 个模型")
    
    print("\n" + "=" * 70)
    print("步骤2: 查找测试文件")
    print("=" * 70)
    
    wav_files = glob.glob('*.wav')
    print(f"Found {len(wav_files)} .wav files to process")
    for f in wav_files:
        print(f"  - {f}")
    
    if len(wav_files) == 0:
        print("\n警告：未找到 .wav 文件，请检查文件路径！")
        return
    
    print("\n" + "=" * 70)
    print("步骤3: 开始批量预测")
    print("=" * 70)
    
    all_results = []
    
    for output_dir in output_directories:
        if not os.path.exists(output_dir):
            print(f"\n跳过不存在的目录: {output_dir}")
            continue
            
        print("\n" + "#" * 70)
        print(f"Processing Output Directory: {os.path.basename(output_dir)}")
        print("#" * 70)
        
        pth_models = get_all_pth_models(output_dir)
        if not pth_models:
            print(f"  未找到 .pth 模型文件，跳过")
            continue
        
        print(f"  发现 {len(pth_models)} 个模型")
        print(f"  预测结果将保存到: {output_dir}")
        
        for model_path, model_name, epoch in pth_models:
            try:
                results = process_single_model(
                    model_path=model_path,
                    model_name=model_name,
                    epoch=epoch,
                    wav_files=wav_files,
                    output_base_dir=output_dir
                )
                all_results.extend(results)
            except Exception as e:
                print(f"\n  处理模型 {model_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "=" * 70)
    print("步骤4: 汇总结果")
    print("=" * 70)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        for output_dir in output_directories:
            if os.path.exists(output_dir):
                summary_path = os.path.join(output_dir, 'all_models_summary.csv')
                summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
                print(f"Summary saved: {summary_path}")
        
        print("\n=== 预测结果汇总 ===")
        display_cols = ['model', 'epoch', 'wav_file', 'accuracy']
        print(summary_df[display_cols].to_string(index=False))
        
        best_idx = summary_df['accuracy'].idxmax()
        best_result = summary_df.loc[best_idx]
        print(f"\n最佳模型: {best_result['model']} (epoch={best_result['epoch']})")
        print(f"最佳准确率: {best_result['accuracy']:.4f}")
        print(f"测试文件: {best_result['wav_file']}")
    else:
        print("没有可汇总的结果")
    
    print("\n" + "=" * 70)
    print("所有模型处理完成！")
    print("=" * 70)


# ======================
# 入口
# ======================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileNet v3 统一入口')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='运行模式: train / predict (默认: train)')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        predict()
    else:
        parser.print_help()
