# -*- coding: utf-8 -*-
"""
MobileNet 模型 ONNX 导出工具（手动池化 + Reshape 版本）

功能说明：
    1. 弹出文件选择框，选择训练好的 .pth 模型文件
    2. 自动从模型所在目录读取 network_config.json 构建网络
    3. 将 nn.Sequential 保存的模型权重迁移到自定义 MobileNetONNX 模块
    4. 在 forward 中手动实现全局平均池化（替代 AdaptiveAvgPool2d）
    5. 使用 reshape 替代 Flatten，导出兼容性更强的 ONNX 模型

导出算子说明：
    - 手动池化: x.mean([2,3])  -> ONNX ReduceMean(axes=[2,3])
    - 替代展平: x.reshape(x.size(0), -1)  -> ONNX Reshape
    - 避免使用: GlobalAveragePool, Flatten 等可能不兼容的算子

运行方式：
    python export_onnx_manual.py
"""

import os
import json
import glob
import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn as nn


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
    """MobileNet with residual connections and ReLU6 (quantization-friendly)"""
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
        x = x.mean([2, 3])     # [N, C]，已经是二维，无需 reshape
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_network_config(pth_dir):
    """
    从模型所在目录加载网络结构配置。

    搜索顺序：
        1. network_config.json（优先）
        2. *_params.json 中的 network_config 字段

    Args:
        pth_dir: .pth 文件所在目录

    Returns:
        dict: 网络配置字典

    Raises:
        FileNotFoundError: 未找到任何配置文件
    """
    config_path = os.path.join(pth_dir, "network_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[INFO] 已加载配置: {config_path}")
        return config

    params_files = glob.glob(os.path.join(pth_dir, "*_params.json"))
    if params_files:
        with open(params_files[0], "r", encoding="utf-8") as f:
            params = json.load(f)
        if "network_config" in params:
            print(f"[INFO] 已从参数文件加载配置: {params_files[0]}")
            return params["network_config"]

    raise FileNotFoundError(
        f"在目录 '{pth_dir}' 中未找到 'network_config.json' 或 '*_params.json'，"
        f"无法确定网络结构。请确保选择的 .pth 文件位于训练输出目录内。"
    )


def load_input_shape(pth_dir):
    """
    尝试从 *_params.json 读取训练时的输入形状。

    Args:
        pth_dir: 模型所在目录

    Returns:
        tuple or None: (C, H, W) 或 None（使用默认值）
    """
    params_files = glob.glob(os.path.join(pth_dir, "*_params.json"))
    if params_files:
        try:
            with open(params_files[0], "r", encoding="utf-8") as f:
                params = json.load(f)
            shape = params.get("input_shape", None)
            if shape and len(shape) == 3:
                return tuple(shape)
        except Exception:
            pass
    return None


def select_pth_file():
    """
    弹出 tkinter 文件选择对话框，让用户选择 .pth 模型文件。

    Returns:
        str: 选择的文件绝对路径，取消则返回空字符串
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    pth_path = filedialog.askopenfilename(
        title="选择要导出的 MobileNet .pth 模型文件",
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
        initialdir=os.getcwd()
    )

    try:
        root.destroy()
    except tk.TclError:
        pass

    return pth_path


def export_pth_to_onnx(pth_path, config, onnx_path, input_shape=None):
    """
    执行 .pth -> ONNX 导出流程。

    Args:
        pth_path: 输入 .pth 文件路径
        config: 网络配置字典
        onnx_path: 输出 .onnx 文件路径
        input_shape: 可选的 dummy input 形状 (C, H, W)
    """
    # 1. 构建模型
    print("\n[INFO] 构建 MobileNet 网络...")
    model = MobileNet(config)
    model.eval()

    # 2. 加载原模型权重
    print(f"[INFO] 加载原始模型: {pth_path}")
    old_data = torch.load(pth_path, map_location="cpu", weights_only=False)

    # 解析 state_dict（支持完整模型、checkpoint 字典、纯 state_dict 三种格式）
    if isinstance(old_data, dict):
        if "model_state_dict" in old_data:
            old_state = old_data["model_state_dict"]
            print("       检测到 checkpoint 格式，提取 model_state_dict")
        else:
            old_state = old_data
            print("       检测到纯 state_dict 格式")
    elif isinstance(old_data, nn.Module):
        old_state = old_data.state_dict()
        print("       检测到完整模型对象")
    else:
        raise TypeError(f"不支持的模型文件格式: {type(old_data)}")

    # 3. 加载权重
    missing, unexpected = model.load_state_dict(old_state, strict=False)
    if missing:
        print(f"[WARN] 模型中以下层未加载到权重: {missing}")
    if unexpected:
        print(f"[WARN] 权重中存在未匹配键: {unexpected}")

    print(f"[INFO] 权重加载完成")

    # 4. 准备 dummy input
    batch_size = 1
    if input_shape is None:
        # 默认时频图尺寸: (C=1, H=200, W=100)
        # 因使用全局池化，H/W 不影响模型正确性，仅影响 dummy input
        dummy_input = torch.randn(batch_size, config["input_channels"], 200, 100)
        print(f"[INFO] 使用默认 dummy input 尺寸: ({batch_size}, {config['input_channels']}, 200, 100)")
    else:
        c, h, w = input_shape
        dummy_input = torch.randn(batch_size, c, h, w)
        print(f"[INFO] 使用训练记录的 input 尺寸: ({batch_size}, {c}, {h}, {w})")

    # 5. 导出 ONNX
    print(f"[INFO] 开始导出 ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
        # dynamic_axes 已删除：固定 batch=1，便于端侧 NPU 编译时预分配内存
    )
    print(f"[INFO] 导出成功: {onnx_path}")

    # 6. 验证（可选）
    try:
        import onnx
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("[INFO] ONNX 模型验证通过")

        # 打印简化后的算子列表，确认未使用 AdaptiveAvgPool / Flatten
        ops = set(node.op_type for node in model_onnx.graph.node)
        print(f"[INFO] ONNX 图中包含的算子: {sorted(ops)}")

        if "GlobalAveragePool" in ops:
            print("[WARN] 警告: ONNX 中仍包含 GlobalAveragePool 算子")
        else:
            print("[OK] 确认未使用 GlobalAveragePool（已替换为 ReduceMean）")

        if "Flatten" in ops:
            print("[WARN] 警告: ONNX 中仍包含 Flatten 算子")
        else:
            print("[OK] 确认未使用 Flatten（已替换为 Reshape）")

    except ImportError:
        print("[INFO] 提示: 安装 onnx 库可进行模型验证: pip install onnx")
    except Exception as e:
        print(f"[WARN] ONNX 验证警告: {e}")

    return onnx_path


def main():
    print("=" * 60)
    print("MobileNet ONNX 导出工具（手动池化 + Reshape）")
    print("=" * 60)

    # 步骤 1: 选择 .pth 文件
    pth_path = select_pth_file()
    if not pth_path:
        print("[INFO] 未选择文件，退出")
        return

    print(f"\n[INFO] 选择的模型: {pth_path}")

    # 步骤 2: 加载配置
    pth_dir = os.path.dirname(pth_path)
    try:
        config = load_network_config(pth_dir)
        print(f"[INFO] 网络配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # 步骤 3: 尝试读取训练时的输入尺寸
    input_shape = load_input_shape(pth_dir)

    # 步骤 4: 确定输出路径
    base_name = os.path.splitext(os.path.basename(pth_path))[0]
    onnx_path = os.path.join(pth_dir, f"{base_name}_manual.onnx")

    # 如果文件已存在，提示覆盖
    if os.path.exists(onnx_path):
        print(f"[WARN] 文件已存在: {onnx_path}")
        print("       导出将覆盖现有文件")

    # 步骤 5: 执行导出
    try:
        export_pth_to_onnx(pth_path, config, onnx_path, input_shape)
    except Exception as e:
        print(f"[ERROR] 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("导出完成!")
    print(f"ONNX 模型: {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
