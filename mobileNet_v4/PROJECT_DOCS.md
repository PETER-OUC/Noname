# 项目文件说明文档

- **生成时间**: 2026-06-04
- **项目路径**: `G:\Files\HW\第三阶段\github\mobileNet_v3`
- **说明**: 本文档基于代码实际实现更新，说明了本项目内各个文件的作用、配置参数及使用方法。

---

## dataprocess_maker.py

### 文件说明

统一数据预处理脚本（可独立运行的 `.py` 版本），无需 Jupyter 环境即可直接执行。将原始 `.wav` 音频文件转换为模型训练所需的时频图 `.npy` 格式。

主要功能包括：
- 批量读取指定文件夹下的 `.wav` 音频文件并执行 STFT（短时傅里叶变换）
- 自动重采样到 48kHz 统一采样率
- **统一使用逐样本减均值 + Min-Max 归一化到 `[0, 1]`，并以 `float16` 存储**
- 支持 SEND 处理（频响曲线叠加）与 5K 样本模式标记
- 支持低频屏蔽（0:40 行置为 `MASK_VALUE`）
- 支持数据增强：基于 `group_d_send` 的频响变换 + 高斯噪声叠加（多 SNR 循环）
- 支持数据集平衡（按比例删减特定标签样本）
- 支持标签合并
- 自动根据配置生成输出文件名后缀
- 输出 `X_train_*.npy`、`X_test_*.npy`、`X_val_*.npy` 及对应的 `y_*.npy`

### 配置参数（在脚本顶部修改）

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `folder_path_list` | 音频文件夹路径列表（按标签顺序） | 字符串列表 |
| `Temp_sample_num` | 预分配样本数上限 | `20000` |
| `Sample_time_len` | 单个样本时长（秒） | `2` |
| `window_length_s` | STFT 窗长（秒） | `0.02` |
| `overlap_ratio` | STFT 窗重叠比例 | `0.0` |
| `overlap_ratio_frame` | 样本帧重叠比例 | `0.5` |
| `frequency_choose_low` / `frequency_choose_high` | 关注频率范围（Hz） | `3e3` / `13e3` |
| `output_channel` | 输出通道数 | `1` |
| `Add_snr` / `Add_snr_1` | 特定标签的额外 SNR 偏移（dB） | `6` / `12` |
| `loop_num` | 每组 group_d 的噪声增强循环次数 | `5` |
| `SNR_dB_begin` | 起始 SNR（dB） | `8` |
| `loss_SNR` | 每次循环 SNR 递减量 | `3` |
| `group_d_loop_num` | group_d_send 组数（当前固定为 3 组） | `3` |
| `NORMALIZATION_TYPE` | 归一化标记（仅影响文件名后缀，实际处理统一为逐样本 Min-Max） | `'global'` / `'column'` |
| `USE_SEND` | 是否启用 SEND 频响叠加 | `True` / `False` |
| `USE_5K` | 是否启用 5K 标记 | `True` / `False` |
| `INCLUDE_ORIGINAL` | 是否保留原始无增强数据 | `True` / `False` |
| `MASK_LOW_FREQ` | 是否屏蔽低频区域（0:40 行） | `True` / `False` |
| `MASK_VALUE` | 低频屏蔽填充值 | `1e-8` |
| `MERGE_LABELS` | 标签合并配置 | `None` 或 `{0: [1]}` |
| `BALANCE_DATASET` | 是否进行数据集平衡 | `True` / `False` |
| `BALANCE_REMOVE_RATIO` | 各标签删除比例 | `{0: 0.2}` |
| `OUTPUT_SUFFIX` | 输出文件名后缀（`None` 时自动根据配置生成） | `None` |

### 使用方法

```bash
python dataprocess_maker.py
```

1. 打开 `dataprocess_maker.py`，在顶部**参数配置区域**修改 `folder_path_list` 和其他参数
2. 直接运行脚本
3. 生成的 `.npy` 文件保存在当前目录，命名格式：`X_train_{Sample_time_len}_{OUTPUT_SUFFIX}.npy`

### 主要函数

- `normalize_per_sample(X)`: 逐样本减均值后 Min-Max 归一化到 `[0, 1]`，支持 2D 单样本或 4D 批量输入
- `smooth_gp_noise(n, std, length_scale)`: 高斯过程平滑扰动，用于生成频响曲线的随机噪声
- `process_batch(X_original, y_original, SNR_dB, group_linear)`: 批量数据增强（加噪声 + 频响变换 + 归一化）
- `process_original_batch(X_original, group_linear)`: 批量处理原始数据（无噪声版本，仅频响变换 + 归一化）
- `balance_dataset(X, y)`: 按比例随机删减指定类别样本以平衡数据集

### 输出文件名后缀生成规则

当 `OUTPUT_SUFFIX` 为 `None` 时，自动按以下规则拼接：
- `5k`（若 `USE_5K=True`）
- `withsend`（若 `USE_SEND=True`）
- `mask` / `nomask`（根据 `MASK_LOW_FREQ`）
- `lieguiyi` / `quanjuguiyi`（根据 `NORMALIZATION_TYPE`）
- `zm_minmax_f16`（固定后缀，表示逐样本 Min-Max + float16）

---

## export_onnx_manual.py

### 文件说明

MobileNet 模型 ONNX 导出工具（手动池化版本）。

该脚本用于将训练好的 `.pth` 模型导出为 `.onnx` 格式，以便部署到端侧设备（如 NPU）。与常规导出不同，本脚本在 `forward` 中手动实现全局平均池化（`x.mean([2,3])`），避免使用 `AdaptiveAvgPool2d` 等可能在端侧不兼容的算子。

### 主要函数

- `load_network_config(pth_dir)`: 从模型所在目录加载 `network_config.json` 或 `*_params.json` 中的 `network_config` 字段
- `load_input_shape(pth_dir)`: 从参数文件中读取训练时的输入形状 `(C, H, W)`
- `select_pth_file()`: 弹出 tkinter 文件选择框，选择 `.pth` 模型文件
- `export_pth_to_onnx(pth_path, config, onnx_path, input_shape)`: 执行 `.pth` -> ONNX 导出流程，固定 `batch_size=1`，`opset_version=10`
- `main()`: 主流程，串联文件选择、配置加载、导出执行与 ONNX 验证

### 网络结构说明

脚本内嵌的 `MobileNet` 类与 `main.py` 中的网络结构一致：
- 使用 `ResidualDSBlock`（Depthwise Separable + 残差连接 + ReLU6）
- `forward` 中：`x.mean([2, 3])` 后直接接 Dropout 和 FC，无需额外展平（已是二维）

### 使用方法

```bash
python export_onnx_manual.py
```

运行后会弹出文件选择框，要求选择一个 `.pth` 模型文件。脚本自动在模型所在目录查找 `network_config.json`，构建网络后导出 ONNX。

导出后的文件命名为 `{原模型名}_manual.onnx`，保存在模型所在目录。导出完成后会进行 ONNX 算子检查，确认未使用 `GlobalAveragePool` 和 `Flatten`。

---

## main.py

### 文件说明

MobileNet 统一入口脚本，合并了训练与预测功能，并移除了对 `d2l` 库的依赖。

**注意**：脚本名称虽含 "v3"，但实际实现为**带残差连接的 Depthwise Separable 自定义轻量网络**（非标准 MobileNetV3）。

支持两种运行模式：
- **train**: 加载 `.npy` 数据，训练模型，自动保存最优模型、断点、ONNX，并自动预测当前目录下的 `.wav` 文件
- **predict**: 批量加载已训练好的 `.pth` 模型，对当前目录下的 `.wav` 文件进行预测，生成混淆矩阵图和 CSV 汇总

### 命令行参数

```bash
python main.py --mode train                      # 训练模式（默认）
python main.py --mode predict                     # 批量预测模式（弹窗/自动发现目录）
python main.py --mode predict --dir <模型目录>     # 直接指定模型目录，跳过弹窗
```

参数说明：
- `--mode`: 运行模式，可选 `train` 或 `predict`，默认为 `train`
- `--dir`: **仅 predict 模式有效**，直接指定模型输出目录路径

### 网络结构定义

网络配置集中管理在 `DEFAULT_NETWORK_CONFIG` 字典中：

```python
DEFAULT_NETWORK_CONFIG = {
    "input_channels": 1,
    "num_classes": 4,
    "dropout_rate": 0.2,
    "stem": {"out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
    "stem_pool": {"kernel_size": 2, "stride": 2, "padding": 0},  # 初始下采样
    "blocks": [
        {"in_channels": 16, "out_channels": 32, "stride": 2},
        {"in_channels": 32, "out_channels": 32, "stride": 1},
        {"in_channels": 32, "out_channels": 64, "stride": 2},
    ],
    "classifier": {"in_features": 64}
}
```

关键组件：
- `ResidualDSBlock`: Depthwise Separable 卷积 + 残差连接 + ReLU6
- `stem_pool`: 初始 MaxPool， aggressive early downsampling
- 全局平均池化：`x.mean([2, 3])`，后接 `reshape` 与 Dropout

### 训练模式（`--mode train`）

#### 数据加载
- 自动扫描当前目录下 `X_tr*.npy`、`y_tr*.npy`、`X_te*.npy`、`y_te*.npy`、`X_va*.npy`、`y_va*.npy`
- 标签转换为 `torch.long`
- 数据形状：`(N, H, W, C)` -> 模型输入 `(C, H, W)`

#### 训练特性
- **断点续训**：启动时检测当前目录下可恢复的断点（`latest_checkpoint.pth`），提示用户选择恢复或新训练
- **自动保存**：每 100 个 epoch 保存检查点，最优模型自动保存为 `{File_name}.pth`
- **混合精度训练**：使用 `torch.amp` 自动混合精度
- **编译优化**：尝试 `torch.compile`（`aot_eager`），失败则回退 eager 模式
- **ONNX 导出**：训练结束后自动导出 FP32 ONNX 模型（含动态 batch）
- **自动预测**：训练完成后，若当前目录有 `.wav` 文件，自动用最优模型预测并保存混淆矩阵

#### 训练配置
- 优化器：SGD（lr=0.005, momentum=0.9）
- 损失函数：CrossEntropyLoss
- 默认 epoch：200
- 默认 batch_size：2048

#### 输出目录
- 新训练：`{File_name}_{timestamp}/`
- 恢复训练：沿用原断点目录
- 输出内容：`.pth` 模型、`.csv` 训练记录、`_params.json` 参数、`.onnx` 模型、混淆矩阵图、断点文件、预处理脚本副本

### 预测模式（`--mode predict`）

#### 预处理参数（硬编码，与训练/数据处理对齐）
| 参数 | 值 | 说明 |
|------|-----|------|
| `Sample_time_len` | `2` | 样本时长 |
| `window_length_s` | `0.02` | STFT 窗长 |
| `overlap_ratio` | `0.0` | STFT 窗重叠 |
| `overlap_ratio_frame` | `0.5` | 样本帧重叠 |
| `frequency_choose_low` / `frequency_choose_high` | `3e3` / `13e3` | 频率范围 |
| `nfft` | `1024` | FFT 点数 |

#### 预测流程
1. 选择/发现模型输出目录
2. 加载目录下所有 `.pth` 模型（排除 `_int8.pth`）
3. 对每个 `.wav` 文件执行 STFT、频响补偿（硬编码 `group_d_send = 140`）、逐样本 Min-Max 归一化
4. 推理后进行**平滑后处理**：若相邻帧预测一致，则修正中间孤立帧
5. 与**硬编码验证标签** `y_val_1s` / `y_val_2s` 对比计算准确率
6. 生成混淆矩阵图（`{wav_name}_best_acc.png`）和 CSV 汇总

#### 硬编码验证标签说明
预测模式内置了固定的验证标签序列 `y_val_1s`（基于 2 秒窗口、0.5 重叠生成），用于与预测结果对比计算准确率。该标签序列按固定模式排列：25 帧目标信号 + 5 帧空白，循环多组。

#### 输出内容
- 混淆矩阵图：`{file_name}_best_acc.png` / `{file_name}_epoch{N}_acc.png`
- 单模型 CSV：`confusion_matrix_{model_name}.csv`
- 汇总 CSV：`all_models_summary.csv`

### 主要函数

- `try_gpu()`: 自动选择 CUDA 或 CPU 设备
- `Accumulator`: 训练指标累加器（替代 d2l）
- `accuracy(y_hat, y)`: 计算分类准确率
- `evaluate_accuracy_gpu(net, data_iter, device)`: GPU 评估模型精度
- `build_mobilenet_from_config(config)`: 根据配置字典构建网络
- `train()`: 训练主函数，含数据加载、断点续训、训练循环、保存、ONNX 导出、验证评估
- `predict(model_dir=None)`: 批量预测主函数，支持弹窗/命令行/自动发现目录
- `process_single_model(...)`: 对单个 `.pth` 模型执行完整预测流程（STFT、归一化、推理、平滑、混淆矩阵、CSV）

### 依赖库

```bash
pip install torch torchvision numpy scipy seaborn pandas scikit-learn matplotlib
```

`torchinfo` 为可选，用于打印模型结构摘要。预测模式需确保当前目录存在待测 `.wav` 文件。
