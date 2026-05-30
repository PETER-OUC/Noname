# 项目文件说明文档

- **生成时间**: 2026-05-30
- **项目路径**: `G:\Files\HW\第三阶段\github\mobileNet_v3`
- **说明**: 本文档详细说明了本项目内各个文件的作用及使用方法。

---

## dataprocess_unified.ipynb

### 文件说明

统一数据预处理 Notebook，用于将原始音频数据（WAV）转换为模型训练所需的时频图 `.npy` 格式。

主要功能包括：
- 读取 `.wav` 音频文件并执行 STFT（短时傅里叶变换）
- 支持列归一化（column）和全局归一化（global）两种方式
- 支持 SEND 处理、5K 样本模式、低频屏蔽等可选数据增强
- 支持标签合并（如将多个类别合并为一个）
- 输出 `X_train_*.npy`、`X_test_*.npy`、`X_val_*.npy` 及对应的 `y_*.npy`

### 配置参数（在 Notebook 内配置单元格中修改）

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `NORMALIZATION_TYPE` | 归一化方式 | `'column'` (列归一), `'global'` (全局归一) |
| `USE_SEND` | 是否使用 SEND 处理 | `True`, `False` |
| `USE_5K` | 是否使用 5K 样本模式 | `True`, `False` |
| `MASK_LOW_FREQ` | 是否屏蔽低频区域 | `True`, `False` |
| `INCLUDE_ORIGINAL` | 是否包含原始数据（无增强） | `True`, `False` |
| `MERGE_LABELS` | 标签合并配置 | 字典或 `None` |
| `OUTPUT_SUFFIX` | 输出文件名后缀 | 字符串 |

### 使用方法

1. 打开 `dataprocess_unified.ipynb`
2. 在第一个代码单元格（参数配置）中修改所需参数
3. 依次运行所有单元格
4. 生成的 `.npy` 文件将保存在当前目录下，供 `main.py` 训练使用

---

## export_onnx_manual.py

### 文件说明

MobileNet 模型 ONNX 导出工具（手动池化 + Reshape 版本）。

该脚本用于将训练好的 `.pth` 模型导出为 `.onnx` 格式，以便部署到端侧设备（如 NPU）。与常规导出不同，本脚本在 forward 中手动实现全局平均池化（`x.mean([2,3])`）和 reshape，避免使用 `AdaptiveAvgPool2d` 和 `Flatten` 等可能在端侧不兼容的算子。

### 主要函数

- `load_network_config(pth_dir)`: 从模型所在目录加载 `network_config.json` 或 `*_params.json`
- `load_input_shape(pth_dir)`: 从参数文件中读取训练时的输入形状
- `select_pth_file()`: 弹出 tkinter 文件选择框，选择 `.pth` 模型文件
- `export_pth_to_onnx(pth_path, config, onnx_path, input_shape)`: 执行 `.pth` -> ONNX 导出流程
- `main()`: 主流程，串联文件选择、配置加载、导出执行

### 使用方法

```bash
python export_onnx_manual.py
```

运行后会弹出文件选择框，要求选择一个 `.pth` 模型文件。脚本会自动在模型所在目录查找 `network_config.json`，构建网络后导出 ONNX。

导出后的文件命名为 `{原模型名}_manual.onnx`，保存在模型所在目录。

---

## main.py

### 文件说明

MobileNet v3 统一入口脚本，合并了原 `mobile_net_train_v3.py`（训练）和 `predict_mobile_net.py`（预测）的功能，并移除了对 `d2l` 库的依赖。

支持两种运行模式：
- **train**: 加载 `.npy` 数据，训练 MobileNet 模型，自动保存最优模型和断点。训练完成后会自动用最优模型预测当前目录下的 `.wav` 文件并保存混淆矩阵。
- **predict**: 批量加载已训练好的 `.pth` 模型，对当前目录下的 `.wav` 文件进行预测并生成混淆矩阵图和 CSV 汇总。

### 命令行参数

```bash
python main.py --mode train      # 训练模式（默认）
python main.py --mode predict    # 批量预测模式
```

参数说明：
- `--mode`: 运行模式，可选 `train` 或 `predict`，默认为 `train`

### 主要函数

- `try_gpu()`: 自动选择 CUDA 或 CPU 设备
- `accuracy(y_hat, y)`: 计算分类准确率
- `evaluate_accuracy_gpu(net, data_iter, device)`: 在 GPU 上评估模型准确率
- `build_mobilenet_from_config(config)`: 根据配置字典构建 MobileNet 网络
- `train()`: 训练主函数，包含数据加载、断点续训、训练循环、模型保存、ONNX 导出、验证集评估、自动预测等完整流程
- `predict()`: 批量预测主函数，支持目录选择/自动发现、多模型批量处理、结果汇总
- `process_single_model(...)`: 对单个 `.pth` 模型执行预测、后处理、混淆矩阵生成和 CSV 保存

### 使用方法

#### 训练

确保当前目录下有训练数据文件（如 `X_train_*.npy`、`y_train_*.npy`、`X_test_*.npy`、`y_test_*.npy`、`X_val_*.npy`、`y_val_*.npy`）：

```bash
python main.py --mode train
```

训练过程中支持：
- **断点续训**: 启动时会检测当前目录下是否有可恢复的训练断点（`latest_checkpoint.pth`），并提示用户选择恢复或重新开始
- **自动保存**: 每 100 个 epoch 保存检查点，最优模型自动保存为 `{File_name}.pth`
- **ONNX 导出**: 训练结束后自动导出 FP32 ONNX 模型
- **自动预测**: 训练完成后，如果当前目录下有 `.wav` 文件，会自动用最优模型进行预测并保存混淆矩阵

训练结果保存在以时间戳命名的输出目录中（如 `mobile_net_xxx_20260530_224724/`）。

#### 批量预测

确保当前目录下有训练输出的模型目录（内含 `.pth` 文件）和 `.wav` 测试文件：

```bash
python main.py --mode predict
```

执行后会：
1. 弹出目录选择框让用户选择模型输出目录（或自动发现）
2. 自动发现目录下的所有 `.pth` 模型（排除 `_int8.pth`）
3. 对每个模型和每个 `.wav` 文件执行预测
4. 生成混淆矩阵图（`{wav_name}_best_acc.png` 或 `{wav_name}_epoch{N}_acc.png`）
5. 生成 `confusion_matrix_{model_name}.csv` 和 `all_models_summary.csv`

### 网络结构配置

网络配置集中管理在 `DEFAULT_NETWORK_CONFIG` 字典中，训练时会自动保存为 `network_config.json`，预测时会自动加载。关键参数包括：

- `input_channels`: 1（单通道灰度图）
- `num_classes`: 4（Nothing, Speedboat, Dolphin, Whale）
- `stem`: 初始卷积层配置（kernel=3, stride=2）
- `blocks`: ResidualDSBlock 列表，定义各阶段的通道数和步长
- `classifier`: 全连接层输入特征数

### 依赖库

运行 `main.py` 需要以下 Python 库：

```bash
pip install torch torchvision numpy scipy seaborn pandas scikit-learn matplotlib
```

（`torchinfo` 为可选，用于打印模型结构摘要）
