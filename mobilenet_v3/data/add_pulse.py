"""
真实脉冲提取与批量注入工具（Python版）
功能：1) Hampel检测真实录音中的碰撞脉冲，建立模板库并统计概率/强度
      2) 遍历仿真数据集路径，按真实统计规律批量注入脉冲
      3) 输出到指定根目录，保留原子文件夹层级
"""

import os
import time
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 参数配置区域 ====================

# ---- 真实海试录音路径（原始含脉冲的未降噪文件） ----
real_wav_path = r'G:\\Files\\HW\\第三阶段\\github\\mobileNet_v3\\pulse_test_3_手表116_level_2_40.wav'

# ---- 真实录音分析时段（秒） ----
analysis_start_sec = 0
analysis_end_sec = np.inf

# ---- 真实录音采样率（Hz） ----
fs_real = 48000

# ---- Hampel参数（与你的降噪代码严格一致） ----
window_size = 200
threshold_coeff = 5.5

# ---- 脉冲聚类参数 ----
min_pulse_gap_ms = 3
extend_margin_ms = 2
local_context_ms = 4

# ---- 仿真数据集源路径（原始干净数据所在位置） ----
folder_path_list = [
    r'G:\\Files\\HW\\第三阶段\\github\\mobileNet_v3\\data\\新数据集\\噪音数据集',
    r'G:\\Files\\HW\\第三阶段\\github\\mobileNet_v3\\data\\新数据集\\快艇',
    r'G:\\Files\\HW\\第三阶段\\github\\mobileNet_v3\\data\\新数据集\\海豚',
    r'G:\\Files\\HW\\第三阶段\\github\\mobileNet_v3\\data\\新数据集\\伪虎鲸'
]

# ---- 注入后输出根目录（自动保留原子文件夹名） ----
output_root = r'G:\\Files\\HW\\第三阶段\\github\\mobileNet_v3\\data\\新数据集-添加脉冲噪声'

# ---- 注入控制 ----
INJECT_ENABLE = True
output_suffix = '_impulse'

# ---- 模板库保存路径 ----
template_npz_path = 'impulse_templates_real.npz'


# ==================== 第一部分：从真实录音提取脉冲并统计 ====================

print('========== 开始提取真实脉冲 ==========')
print(f'读取: {real_wav_path}')

data_real, fs_read = sf.read(real_wav_path)
if fs_read != fs_real:
    raise ValueError(f'录音实际采样率 {fs_read} Hz 与设定值 {fs_real} Hz 不匹配，请修改 fs_real')

if data_real.ndim > 1:
    data_real = data_real[:, 0]
data_real = np.asarray(data_real).ravel()

total_dur_real = len(data_real) / fs_real
s1 = max(0, int(round(analysis_start_sec * fs_real)))
if np.isinf(analysis_end_sec):
    s2 = len(data_real)
    analysis_end_sec = total_dur_real
else:
    s2 = min(len(data_real), int(round(analysis_end_sec * fs_real)))

data_seg = data_real[s1:s2]
analysis_dur_sec = (s2 - s1) / fs_real

print(f'分析时段: {s1/fs_real:.3f} s ~ {s2/fs_real:.3f} s (时长 {analysis_dur_sec:.3f} s)')

n = len(data_seg)
half_window = window_size // 2
is_outlier = np.zeros(n, dtype=bool)

print(f'Hampel检测中 (共 {n} 点, 窗口 {window_size})...')
t0 = time.time()
for i in range(n):
    idx_s = max(0, i - half_window)
    idx_e = min(n, i + half_window + 1)  # Python切片不含终点，+1
    w = data_seg[idx_s:idx_e]
    med = np.median(w)
    mad_val = np.median(np.abs(w - med)) * 1.4826
    if np.abs(data_seg[i] - med) > threshold_coeff * mad_val:
        is_outlier[i] = True
elapsed = time.time() - t0
outlier_count = int(np.sum(is_outlier))
print(f'Hampel完成: 耗时 {elapsed:.1f} 秒, 标记 {outlier_count} 个异常点 ({100*outlier_count/n:.4f}%)')

outlier_idx = np.where(is_outlier)[0]
if outlier_idx.size == 0:
    raise ValueError('未检测到任何脉冲，请检查录音路径、时段或Hampel参数')

gaps = np.diff(outlier_idx)
gap_thresh_samp = max(1, int(round(fs_real * min_pulse_gap_ms / 1000)))
breaks = np.where(gaps > gap_thresh_samp)[0]

pulse_starts = outlier_idx[np.concatenate(([0], breaks + 1))]
pulse_ends = outlier_idx[np.concatenate((breaks, [len(outlier_idx) - 1]))]
num_pulses = len(pulse_starts)

print(f'聚类完成: 共 {num_pulses} 个脉冲事件 (间隔阈值 {gap_thresh_samp} 点)')

ext_samp = max(1, int(round(fs_real * extend_margin_ms / 1000)))
local_samp = max(window_size, int(round(fs_real * local_context_ms / 1000)))

pulse_templates = []
pulse_dur_ms = np.zeros(num_pulses)
pulse_peak_ratio = np.zeros(num_pulses)
pulse_rms_ratio = np.zeros(num_pulses)
pulse_energy_ratio = np.zeros(num_pulses)

for p in range(num_pulses):
    o_start = pulse_starts[p]
    o_end = pulse_ends[p]
    dur_samp = o_end - o_start + 1
    
    ext1 = max(0, o_start - ext_samp)
    ext2 = min(n, o_end + ext_samp + 1)
    waveform = data_seg[ext1:ext2].copy()
    
    ctx1 = max(0, o_start - local_samp)
    ctx2 = min(n, o_end + local_samp + 1)
    local_rms = np.sqrt(np.mean(data_seg[ctx1:ctx2]**2))
    
    peak_amp = np.max(np.abs(waveform))
    pulse_rms = np.sqrt(np.mean(waveform**2))
    
    pulse_templates.append(waveform)
    pulse_dur_ms[p] = dur_samp / fs_real * 1000
    pulse_peak_ratio[p] = peak_amp / (local_rms + np.finfo(float).eps)
    pulse_rms_ratio[p] = pulse_rms / (local_rms + np.finfo(float).eps)
    pulse_energy_ratio[p] = (pulse_rms / (local_rms + np.finfo(float).eps))**2

avg_pulses_per_sec = num_pulses / analysis_dur_sec

print('\n========== 真实脉冲统计结果 ==========')
print(f'分析时长:          {analysis_dur_sec:.3f} 秒')
print(f'脉冲总数:          {num_pulses}')
print(f'每秒脉冲数:        {avg_pulses_per_sec:.4f} (约每 {1/(avg_pulses_per_sec+np.finfo(float).eps):.1f} 秒 1 个)')
print(f'脉冲持续时间:      {np.min(pulse_dur_ms):.3f} ~ {np.max(pulse_dur_ms):.3f} ms (均值 {np.mean(pulse_dur_ms):.3f})')
print(f'峰值/局部RMS比:    {np.min(pulse_peak_ratio):.2f} ~ {np.max(pulse_peak_ratio):.2f} (均值 {np.mean(pulse_peak_ratio):.2f})')
print(f'RMS/局部RMS比:     {np.min(pulse_rms_ratio):.2f} ~ {np.max(pulse_rms_ratio):.2f} (均值 {np.mean(pulse_rms_ratio):.2f})')
print(f'能量密度比:        {np.min(pulse_energy_ratio):.2f} ~ {np.max(pulse_energy_ratio):.2f} (均值 {np.mean(pulse_energy_ratio):.2f})')

# 保存模板库（npz格式）
np.savez(template_npz_path,
         pulse_templates=pulse_templates,
         pulse_dur_ms=pulse_dur_ms,
         pulse_peak_ratio=pulse_peak_ratio,
         pulse_rms_ratio=pulse_rms_ratio,
         pulse_energy_ratio=pulse_energy_ratio,
         avg_pulses_per_sec=avg_pulses_per_sec,
         fs_real=fs_real,
         num_pulses=num_pulses,
         analysis_dur_sec=analysis_dur_sec,
         window_size=window_size,
         threshold_coeff=threshold_coeff)
print(f'\n模板库已保存: {template_npz_path}')

# 绘制真实脉冲模板示例
num_plot = min(6, num_pulses)
fig, axes = plt.subplots(num_plot, 1, figsize=(10, 2*num_plot), sharex=False)
if num_plot == 1:
    axes = [axes]

for k in range(num_plot):
    ax = axes[k]
    templ = pulse_templates[k]
    t_ms = np.arange(len(templ)) / fs_real * 1000
    ax.plot(t_ms, templ, 'b')
    
    # 计算异常起点/终点在截取波形中的相对位置
    rel_start = (pulse_starts[k] - max(0, pulse_starts[k] - ext_samp)) / fs_real * 1000
    rel_end = (pulse_ends[k] - max(0, pulse_starts[k] - ext_samp)) / fs_real * 1000
    ax.axvline(rel_start, color='r', linestyle='--', label='异常起点')
    ax.axvline(rel_end, color='r', linestyle='--', label='异常终点')
    
    ax.set_title(f'模板#{k+1}: {pulse_dur_ms[k]:.2f} ms, 峰值比 {pulse_peak_ratio[k]:.1f}, RMS比 {pulse_rms_ratio[k]:.1f}')
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('幅度')
    ax.grid(True)

plt.tight_layout()
plt.show()


# ==================== 第二部分：向仿真数据集批量注入 ====================
if not INJECT_ENABLE:
    print('\nINJECT_ENABLE = False，跳过注入，脚本结束')
    exit()

print('\n========== 开始批量注入仿真数据集 ==========')

# 确保输出根目录存在
os.makedirs(output_root, exist_ok=True)
print(f'创建输出根目录: {output_root}')

for folder_idx, src_folder in enumerate(folder_path_list, start=1):
    if not os.path.isdir(src_folder):
        print(f'警告: 源路径不存在，跳过: {src_folder}')
        continue
    
    # 提取原子文件夹名（如 "噪音数据集"）
    sub_name = os.path.basename(src_folder)
    
    # 构造输出子文件夹路径
    out_subfolder = os.path.join(output_root, sub_name)
    os.makedirs(out_subfolder, exist_ok=True)
    print(f'创建输出子目录: {out_subfolder}')
    
    wav_list = [f for f in os.listdir(src_folder) if f.lower().endswith('.wav')]
    n_files = len(wav_list)
    print(f'\n[{folder_idx}/{len(folder_path_list)}] 源文件夹: {src_folder} -> 输出到: {out_subfolder} (共 {n_files} 个文件)')
    
    for f_idx, wav_name in enumerate(wav_list, start=1):
        wav_path = os.path.join(src_folder, wav_name)
        
        try:
            sim_data, fs_sim = sf.read(wav_path)
        except Exception as e:
            print(f'  读取失败 {wav_name}: {e}')
            continue
            
        if fs_sim != fs_real:
            print(f'  跳过 {wav_name} (采样率 {fs_sim} != {fs_real})')
            continue
        if sim_data.ndim > 1:
            sim_data = sim_data[:, 0]
        sim_data = np.asarray(sim_data).ravel()
        
        sim_dur_sec = len(sim_data) / fs_sim
        sim_rms = np.sqrt(np.mean(sim_data**2))
        
        lambda_val = avg_pulses_per_sec * sim_dur_sec
        num_inject = np.random.poisson(lambda_val)
        
        if num_inject > 0:
            for _ in range(num_inject):
                idx_templ = np.random.randint(0, num_pulses)
                templ = pulse_templates[idx_templ]
                templ_rms = np.sqrt(np.mean(templ**2))
                
                target_rms_ratio = pulse_rms_ratio[np.random.randint(0, num_pulses)]
                target_rms_ratio = target_rms_ratio * (0.8 + 0.4 * np.random.rand())
                
                if templ_rms < 1e-12 or sim_rms < 1e-12:
                    continue
                scale = (sim_rms * target_rms_ratio) / templ_rms
                
                if np.random.rand() > 0.5:
                    templ = -templ
                
                Lp = len(templ)
                if Lp >= len(sim_data):
                    continue
                pos = np.random.randint(0, len(sim_data) - Lp + 1)
                
                sim_data[pos:pos+Lp] = sim_data[pos:pos+Lp] + templ * scale
            
            max_val = np.max(np.abs(sim_data))
            if max_val > 1.0:
                print(f'    警告 {wav_name} 峰值 {max_val:.3f} 超限，已截断')
                sim_data = np.clip(sim_data, -1.0, 1.0)
            
            print(f'  [{f_idx}/{n_files}] {wav_name}: 注入 {num_inject} 个脉冲 (λ={lambda_val:.2f}), 时长 {sim_dur_sec:.2f} s')
        else:
            print(f'  [{f_idx}/{n_files}] {wav_name}: 未注入 (λ={lambda_val:.2f}), 时长 {sim_dur_sec:.2f} s')
        
        # 输出到新路径
        name_only, ext = os.path.splitext(wav_name)
        out_name = f'{name_only}{output_suffix}{ext}'
        out_path = os.path.join(out_subfolder, out_name)
        sf.write(out_path, sim_data, fs_sim)

print('\n所有文件处理完成！')
print(f'输出根目录: {output_root}')
print('子文件夹结构与原数据集保持一致')
print('下一步: 将Python预处理代码中的 folder_path_list 指向新目录即可')
