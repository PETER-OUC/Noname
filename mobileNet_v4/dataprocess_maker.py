# ==========================================
# 参数配置区域 - 在此修改所有配置参数
# ==========================================

# ---- 数据路径配置 ----
# folder_path_list = [
#     "G:\Files\HW\第三阶段\github\mobileNet_v3\data\\新数据集-添加脉冲噪声\\噪音数据集",
#     "G:\Files\HW\第三阶段\github\mobileNet_v3\data\\新数据集-添加脉冲噪声\\快艇",  
#     "G:\Files\HW\第三阶段\github\mobileNet_v3\data\\新数据集-添加脉冲噪声\\海豚", 
#     "G:\Files\HW\第三阶段\github\mobileNet_v3\data\\新数据集-添加脉冲噪声\\伪虎鲸"
# ]
folder_path_list = [
    "G:\\Files\\HW\\第三阶段\\新数据集-添加脉冲噪声\\噪音数据集",
    "G:\\Files\\HW\\第三阶段\\新数据集-添加脉冲噪声\\快艇",  
    "G:\\Files\\HW\\第三阶段\\新数据集-添加脉冲噪声\\海豚", 
    "G:\\Files\\HW\\第三阶段\\新数据集-添加脉冲噪声\\伪虎鲸"

]
# ---- 基础参数配置 ----
Temp_sample_num = 20000
Sample_time_len = 2
window_length_s = 0.02
overlap_ratio = 0.0
overlap_ratio_frame = 0.5
frequency_choose_low = 3e3
frequency_choose_high = 13e3
output_channel = 1

# ---- 数据增强参数配置 ----
Add_snr = 6
Add_snr_1 = 12
loop_num = 5
SNR_dB_begin = 8
loss_SNR = 3
group_d_loop_num = 3


# ---- 处理方式配置 ----
NORMALIZATION_TYPE = 'global'
USE_SEND = False
USE_5K = False
INCLUDE_ORIGINAL = False

# ---- 频段处理配置 ----
MASK_LOW_FREQ = False
MASK_VALUE = 1e-8

# ---- 标签合并配置 ----
MERGE_LABELS = None

# ---- 输出文件名配置 ----
OUTPUT_SUFFIX = None

# ---- 数据集平衡配置 ----
BALANCE_DATASET = True
BALANCE_REMOVE_RATIO = {0: 0.2}

# ==========================================
# 自动生成输出文件名后缀
# ==========================================
if OUTPUT_SUFFIX is None:
    parts = []
    if USE_5K:
        parts.append('5k')
    if USE_SEND:
        parts.append('withsend')
    if MASK_LOW_FREQ:
        parts.append('mask')
    else:
        parts.append('nomask')
    if NORMALIZATION_TYPE == 'column':
        parts.append('lieguiyi')
    elif NORMALIZATION_TYPE == 'global':
        parts.append('quanjuguiyi')
    parts.append('zm_minmax_f16')
    OUTPUT_SUFFIX = '_'.join(parts) if parts else 'standard'

print(f"配置完成！输出文件名后缀: {OUTPUT_SUFFIX}")
print(f"归一化方式: 逐样本减均值 + Min-Max归一化到[0,1] (float16存储)")
print(f"使用send处理: {USE_SEND}")
print(f"5k模式: {USE_5K}")
print(f"低频屏蔽: {MASK_LOW_FREQ}")
print(f"包含原始数据: {INCLUDE_ORIGINAL}")

# ==========================================
# 导入必要的库
# ==========================================
import os
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.signal as signal
from scipy.io import wavfile
from scipy import interpolate
import matplotlib.pyplot as plt
import gc

# ==========================================
# 统计文件数量
# ==========================================
files_all_num = 0
for folder_path in folder_path_list:
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    num_files = len(wav_files)
    files_all_num = files_all_num + num_files
    print(f'{folder_path}文件夹下，共有 {num_files} 个文件。')

# ==========================================
# 【修改】归一化函数：逐样本减均值 + Min-Max归一化到[0,1]
# ==========================================
def normalize_per_sample(X):
    """
    逐样本：减均值后，Min-Max归一化到[0, 1]。
    支持2D单样本或4D批量输入。
    """
    X = np.asarray(X)
    if X.ndim == 2:
        X = X - X.mean()
        X_min, X_max = X.min(), X.max()
        if X_max > X_min:
            return (X - X_min) / (X_max - X_min)
        return np.zeros_like(X)
    elif X.ndim == 4:
        out = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            for c in range(X.shape[3]):
                x = X[i, :, :, c]
                x = x - x.mean()
                x_min, x_max = x.min(), x.max()
                if x_max > x_min:
                    out[i, :, :, c] = (x - x_min) / (x_max - x_min)
                else:
                    out[i, :, :, c] = 0.0
        return out
    else:
        raise ValueError("输入必须是2D或4D数组")


# ==========================================
# 平滑随机扰动
# ==========================================
def smooth_gp_noise(n, std, length_scale=0.05):
    x = np.linspace(0, 1, n)
    dist_sq = ((x[:, None] - x[None, :]) / length_scale) ** 2
    K = np.exp(-0.5 * dist_sq) + 1e-6 * np.eye(n)
    L = np.linalg.cholesky(K)
    z = np.random.randn(n)
    y = L @ z
    if np.std(y) > 1e-9:
        y = y / np.std(y) * std
    return y

# ==========================================
# 音频数据处理和STFT转换
# ==========================================
nfft = 1024
if overlap_ratio == 0:
    time_bins = np.floor(Sample_time_len / window_length_s)
else:
    time_bins = np.floor((Sample_time_len - window_length_s) / (window_length_s * (1 - overlap_ratio))) + 2

freq_bins = np.floor((frequency_choose_high - frequency_choose_low) * (nfft/48000))
time_bins, freq_bins = int(time_bins), int(freq_bins)
print(f'时频矩阵维度：时间轴 {time_bins}，频率轴 {freq_bins}')

# 预分配改为 float16
data_matrix = np.zeros((Temp_sample_num, freq_bins, time_bins, output_channel), dtype=np.float16)
label_matrix = np.zeros((Temp_sample_num, 1), dtype=np.uint8)
signal_len = np.zeros((files_all_num, 4))

Time_end_index = time_bins
sample_idex = 0
label_idex = 0
all_time = 0
temp_1 = 0
signal_one = 0
file_one = -1

for folder_path in folder_path_list:
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    num_files = len(wav_files)
    file_one = file_one + 1 
    
    for idx, wav_file in enumerate(wav_files):
        file_path = os.path.join(folder_path, wav_file)
        Fs, audio_data = wavfile.read(file_path)
        target_Fs = 48000
        num_samples = int(len(audio_data) * target_Fs / Fs)
        audio_data = signal.resample(audio_data, num_samples)
        Fs = target_Fs

        all_time = all_time + len(audio_data)/Fs
        signal_duration = len(audio_data) / Fs
        sample_step = Sample_time_len * (1 - overlap_ratio_frame)
        frame_num = np.floor((signal_duration - Sample_time_len) / sample_step) + 1
        
        signal_len[signal_one, file_one] = signal_duration
        signal_one = signal_one + 1

        if frame_num >= 1:
            window_len = int(window_length_s * Fs)
            window_overlap = int(overlap_ratio * window_len)
            nfft = 1024 
            f, t, Zxx = signal.stft(audio_data, fs=Fs, window='hamming', 
                            nperseg=window_len, noverlap=window_overlap, nfft=nfft)
            Frequency_start_index = np.argmin(np.abs(f - (frequency_choose_low)))
            Frequency_end_index = Frequency_start_index + freq_bins

            stft_time_resolution = t[1] - t[0]
            samples_per_frame = int(Sample_time_len / stft_time_resolution)
            
            for frame_one in range(int(frame_num)):
                start_time = frame_one * sample_step
                end_time = start_time + Sample_time_len
                
                start_idx = np.argmin(np.abs(t - start_time))
                end_idx = start_idx + samples_per_frame
                
                if end_idx <= len(t):
                    X_temp = np.abs(Zxx[Frequency_start_index:Frequency_end_index, 
                                      start_idx:end_idx])
                    
                    if MASK_LOW_FREQ:
                        X_temp[0:40, :] = MASK_VALUE
                    
                    if X_temp.shape[1] < time_bins:
                        padding = time_bins - X_temp.shape[1]
                        X_temp = np.pad(X_temp, ((0, 0), (0, padding)), mode='constant')
                    
                    # 【修改】逐样本：减均值 + Min-Max到[0,1]，再存float16
                    if sample_idex < Temp_sample_num:
                        X_temp = X_temp.astype(np.float32)
                        X_temp = X_temp - X_temp.mean()
                        X_min, X_max = X_temp.min(), X_temp.max()
                        if X_max > X_min:
                            X_temp = (X_temp - X_min) / (X_max - X_min)
                        else:
                            X_temp = np.zeros_like(X_temp)
                        data_matrix[sample_idex, :, :, 0] = X_temp.astype(np.float16)
                        label_matrix[sample_idex, 0] = label_idex
                        sample_idex += 1
    
    print(f'{folder_path}处理完成，共处理 {sample_idex-temp_1} 个样本。')
    temp_1 = sample_idex
    label_idex += 1
    label_matrix = label_matrix.astype(np.uint8)

# 裁剪
data_matrix = data_matrix[0:sample_idex, :, :, :]
label_matrix = label_matrix[0:sample_idex, :]
print(f'处理完成，共处理 {sample_idex} 个样本。')

# 分割数据集
X_train, X_test_val, y_train, y_test_val = train_test_split(
    data_matrix, label_matrix, test_size=0.3, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_test_val, y_test_val, test_size=0.8, random_state=1)

y_train_1 = y_train.ravel()
y_val_1 = y_val.ravel()
y_test_1 = y_test.ravel()

X_train_original = X_train.copy()
X_test_original = X_test.copy()
X_val_original = X_val.copy()

# ==========================================
# group_d_send 定义
# ==========================================
group_d_send_1 = np.array([
    -195.9427033, -190.6943824, -187.5827526, -191.0283226, -192.8720064,
    -195.4116142, -193.7250014, -192.3906152, -190.4235213, -189.5295085,
    -193.7564105, -198.7959695, -201.4739388, -202.1151833, -204.8464573,
    -208.6816811, -206.1343964, -205.8351998, -204.4214199, -206.7228647,
    -213.7277083, -213.5398487, -207.3456817, -202.2751831, -199.8710637,
    -197.9323103, -196.6424151, -195.9883483, -195.9012038, -194.5879478,
    -192.1384018, -191.3194873, -193.0034583, -196.1234345, -195.0501658,
    -193.1474387, -191.5715275, -190.8317177
])

group_d_send_2 = np.array([
    -190.1431136, -192.7725689, -194.9442765, -195.8188706, -202.1549129,
    -197.2489038, -194.0295998, -192.0832959, -189.8711661, -189.5390709,
    -195.4889969, -200.1281825, -200.5610529, -202.0359043, -205.076686,
    -204.1000371, -206.6412282, -209.0587711, -208.6208275, -216.1524201,
    -221.9369258, -209.3926208, -202.2351173, -198.902799, -195.8928552,
    -193.7635142, -192.3231952, -191.4257878, -192.9530334, -195.5375161,
    -195.1620206, -193.745072, -192.7121873, -192.7886533, -193.1139023,
    -192.0614355, -191.5179716, -191.0386636
])

group_d_send_3 = np.array([
    -189.3997016, -187.3735325, -188.2503699, -190.5967936, -191.0913439,
    -204.5899975, -198.6195831, -195.1442834, -193.3941964, -192.4280649,
    -195.6468349, -198.4285022, -201.3400561, -203.3667053, -201.2694447,
    -201.5910855, -208.1234977, -212.0228809, -208.6584484, -205.1266813,
    -208.5184645, -225.9952472, -209.0172702, -205.7473891, -202.712071,
    -199.7725648, -198.2208079, -196.7826835, -195.1535732, -194.8565119,
    -193.1574725, -191.5841589, -191.3811628, -192.5032542, -192.7531798,
    -191.8220394, -191.2388098, -190.8169933
])

group_d_list = [group_d_send_1, group_d_send_2, group_d_send_3]

if USE_SEND:
    group_d_send_base = np.array([
        135, 136,136,136,136,137, 137,137,137,137,138,
        138,138,138,138,139, 139,139,139,139,140,
        140,140,140,140,141, 141,141,141,141,142,
        142,142,142,142,143, 143,143,143,143,144,
        144,144,144,144,145, 145,145,145,145,146,
        146,146,146,146,147, 147,147,147,147,148,
        148,148,148,148,149, 149,149,149,149,150,
        150,150,150,150,150, 150,150,150,150,150
    ])
    freq_range_original_send = np.linspace(3e3, 11e3, len(group_d_send_base))
    f_interp_send = interpolate.interp1d(
        freq_range_original_send, group_d_send_base, kind='cubic', fill_value='extrapolate')
    print("已启用send处理")
else:
    print("未启用send处理")

# ==========================================
# 数据增强处理
# ==========================================
freq_range_original = np.linspace(2500, 21000, len(group_d_send_1))
freq_range_target = np.linspace(
    frequency_choose_low, frequency_choose_high - int(1/window_length_s), freq_bins)

train_samples = X_train_original.shape[0]
val_samples = X_val_original.shape[0]
test_samples = X_test_original.shape[0]
sample_shape = X_train_original.shape[1:]

if INCLUDE_ORIGINAL:
    total_train_samples = train_samples * (group_d_loop_num * (loop_num + 1) + 1)
    total_val_samples = val_samples * (group_d_loop_num * (loop_num + 1) + 1)
    total_test_samples = test_samples * (group_d_loop_num * (loop_num + 1) + 1)
else:
    total_train_samples = train_samples * (group_d_loop_num * (loop_num + 1))
    total_val_samples = val_samples * (group_d_loop_num * (loop_num + 1))
    total_test_samples = test_samples * (group_d_loop_num * (loop_num + 1))

# 预分配全部改为 float16
X_train = np.zeros((total_train_samples, *sample_shape), dtype=np.float16)
y_train = np.zeros(total_train_samples, dtype=y_train_1.dtype)
X_val = np.zeros((total_val_samples, *sample_shape), dtype=np.float16)
y_val = np.zeros(total_val_samples, dtype=y_val_1.dtype)
X_test = np.zeros((total_test_samples, *sample_shape), dtype=np.float16)
y_test = np.zeros(total_test_samples, dtype=y_test_1.dtype)

# 【修改】批量处理函数：逐样本减均值 + Min-Max到[0,1]
def process_batch(X_original, y_original, SNR_dB, group_linear):
    batch_size = X_original.shape[0]
    result = np.zeros((batch_size, *sample_shape), dtype=np.float16)
    
    for i in range(batch_size):
        # 用float32做所有计算，避免float16精度爆炸
        X_temp = X_original[i, :, :, 0].astype(np.float32)
        signal_power = np.mean(X_temp ** 2)
        
        if y_original[i] == 1:
            noise_power = signal_power / (10 ** ((SNR_dB + Add_snr) / 10))
        elif y_original[i] == 2:
            noise_power = signal_power / (10 ** ((SNR_dB + Add_snr_1) / 10))
        else:
            noise_power = signal_power / (10 ** (SNR_dB / 10))
        
        noise = np.random.normal(0, np.sqrt(noise_power), X_temp.shape)
        X_temp = X_temp + noise
        X_temp = X_temp * group_linear
        
        if MASK_LOW_FREQ:
            X_temp[0:40, :] = MASK_VALUE
        
        # 【修改】逐样本：减均值 + Min-Max归一化到[0,1]
        X_temp = X_temp - X_temp.mean()
        X_min, X_max = X_temp.min(), X_temp.max()
        if X_max > X_min:
            X_temp = (X_temp - X_min) / (X_max - X_min)
        else:
            X_temp = np.zeros_like(X_temp)
        
        # 转float16写入
        result[i, :, :, 0] = X_temp.astype(np.float16)
    
    return result

def process_original_batch(X_original, group_linear):
    batch_size = X_original.shape[0]
    result = np.zeros((batch_size, *sample_shape), dtype=np.float16)
    
    for i in range(batch_size):
        X_temp = X_original[i, :, :, 0].astype(np.float32)
        X_temp = X_temp * group_linear
        
        if MASK_LOW_FREQ:
            X_temp[0:40, :] = MASK_VALUE
        
        # 【修改】逐样本：减均值 + Min-Max归一化到[0,1]
        X_temp = X_temp - X_temp.mean()
        X_min, X_max = X_temp.min(), X_temp.max()
        if X_max > X_min:
            X_temp = (X_temp - X_min) / (X_max - X_min)
        else:
            X_temp = np.zeros_like(X_temp)
        
        result[i, :, :, 0] = X_temp.astype(np.float16)
    
    return result

# 主循环
current_train_idx = 0
current_val_idx = 0
current_test_idx = 0

# 【修改】INCLUDE_ORIGINAL=True 时，同样改为逐样本减均值 + Min-Max到[0,1]
if INCLUDE_ORIGINAL:
    for i in range(train_samples):
        X_temp = X_train_original[i, :, :, 0].astype(np.float32)
        X_temp = X_temp - X_temp.mean()
        X_min, X_max = X_temp.min(), X_temp.max()
        if X_max > X_min:
            X_temp = (X_temp - X_min) / (X_max - X_min)
        else:
            X_temp = np.zeros_like(X_temp)
        X_train[i, :, :, 0] = X_temp.astype(np.float16)
    y_train[:train_samples] = y_train_1
    current_train_idx = train_samples

    for i in range(val_samples):
        X_temp = X_val_original[i, :, :, 0].astype(np.float32)
        X_temp = X_temp - X_temp.mean()
        X_min, X_max = X_temp.min(), X_temp.max()
        if X_max > X_min:
            X_temp = (X_temp - X_min) / (X_max - X_min)
        else:
            X_temp = np.zeros_like(X_temp)
        X_val[i, :, :, 0] = X_temp.astype(np.float16)
    y_val[:val_samples] = y_val_1
    current_val_idx = val_samples

    for i in range(test_samples):
        X_temp = X_test_original[i, :, :, 0].astype(np.float32)
        X_temp = X_temp - X_temp.mean()
        X_min, X_max = X_temp.min(), X_temp.max()
        if X_max > X_min:
            X_temp = (X_temp - X_min) / (X_max - X_min)
        else:
            X_temp = np.zeros_like(X_temp)
        X_test[i, :, :, 0] = X_temp.astype(np.float16)
    y_test[:test_samples] = y_test_1
    current_test_idx = test_samples

for group_idx, group_d in enumerate(group_d_list):
    print(f"\n处理第 {group_idx+1} 个 group_d_send...")
    
    group_d_std = np.std(group_d)
    f_interp = interpolate.interp1d(
        freq_range_original, group_d, kind='cubic', fill_value='extrapolate')
    group_d_interp_into = f_interp(freq_range_target)
    
    if USE_SEND:
        group_d_interp_send = f_interp_send(freq_range_target)
        group_d_interp = group_d_interp_into + group_d_interp_send
    else:
        group_d_interp = group_d_interp_into
    
    groub_list = np.zeros((loop_num, freq_bins))
    for loop_one in range(loop_num):
        noise = smooth_gp_noise(freq_bins, group_d_std * 2, length_scale=0.05)
        groub_list[loop_one, :] = group_d_interp + noise
    
    group_d_interp_reshaped = group_d_interp.reshape(-1, 1)
    group_linear_original = 10 ** ((group_d_interp_reshaped - np.mean(group_d_interp_reshaped)) / 20)
    
    end_train_idx = current_train_idx + train_samples
    X_train[current_train_idx:end_train_idx] = process_original_batch(
        X_train_original, group_linear_original)
    y_train[current_train_idx:end_train_idx] = y_train_1
    current_train_idx = end_train_idx
    
    end_val_idx = current_val_idx + val_samples
    X_val[current_val_idx:end_val_idx] = process_original_batch(
        X_val_original, group_linear_original)
    y_val[current_val_idx:end_val_idx] = y_val_1
    current_val_idx = end_val_idx
    
    end_test_idx = current_test_idx + test_samples
    X_test[current_test_idx:end_test_idx] = process_original_batch(
        X_test_original, group_linear_original)
    y_test[current_test_idx:end_test_idx] = y_test_1
    current_test_idx = end_test_idx
    
    SNR_dB_start = SNR_dB_begin
    for loop_one in range(loop_num):
        SNR_dB = SNR_dB_start - loop_one * loss_SNR
        
        group = groub_list[loop_one, :].reshape(-1, 1)
        group_linear = 10 ** ((group - np.mean(group)) / 20)
        
        end_train_idx = current_train_idx + train_samples
        X_train[current_train_idx:end_train_idx] = process_batch(
            X_train_original, y_train_1, SNR_dB, group_linear)
        y_train[current_train_idx:end_train_idx] = y_train_1
        current_train_idx = end_train_idx
        
        end_val_idx = current_val_idx + val_samples
        X_val[current_val_idx:end_val_idx] = process_batch(
            X_val_original, y_val_1, SNR_dB, group_linear)
        y_val[current_val_idx:end_val_idx] = y_val_1
        current_val_idx = end_val_idx
        
        end_test_idx = current_test_idx + test_samples
        X_test[current_test_idx:end_test_idx] = process_batch(
            X_test_original, y_test_1, SNR_dB, group_linear)
        y_test[current_test_idx:end_test_idx] = y_test_1
        current_test_idx = end_test_idx
        
        if loop_one % 10 == 0:
            gc.collect()
    
    print(f"第 {group_idx+1} 个 group_d_send 处理完成，已生成 {loop_num+1} 个增强版本")

print(f"\n所有group_d_send处理完成！")

# ==========================================
# 数据集裁剪和平衡
# ==========================================
X_train = X_train[:current_train_idx]
y_train = y_train[:current_train_idx]
X_val = X_val[:current_val_idx]
y_val = y_val[:current_val_idx]
X_test = X_test[:current_test_idx]
y_test = y_test[:current_test_idx]

print(f"\n数据增强完成！")
print(f"训练集: {X_train.shape[0]} 个样本")
print(f"验证集: {X_val.shape[0]} 个样本")
print(f"测试集: {X_test.shape[0]} 个样本")

# 数据集平衡
if BALANCE_DATASET:
    def balance_dataset(X, y):
        y = y.ravel()
        keep_mask = np.ones(len(y), dtype=bool)
        for label, ratio in BALANCE_REMOVE_RATIO.items():
            indices = np.where(y == label)[0]
            if len(indices) > 0:
                np.random.seed(42)
                remove_indices = np.random.choice(
                    indices, size=int(len(indices) * ratio), replace=False)
                keep_mask[remove_indices] = False
        return X[keep_mask], y[keep_mask]

    print("\n开始平衡数据集...")
    X_train, y_train = balance_dataset(X_train, y_train)
    X_val, y_val = balance_dataset(X_val, y_val)
    X_test, y_test = balance_dataset(X_test, y_test)

    print(f"平衡后训练集: {X_train.shape[0]} 个样本")
    print(f"平衡后验证集: {X_val.shape[0]} 个样本")
    print(f"平衡后测试集: {X_test.shape[0]} 个样本")

gc.collect()

# ==========================================
# 标签合并和保存
# ==========================================
if MERGE_LABELS is not None:
    print("\n进行标签合并...")
    for target_label, source_labels in MERGE_LABELS.items():
        for source_label in source_labels:
            y_train[y_train == source_label] = target_label
            y_val[y_val == source_label] = target_label
            y_test[y_test == source_label] = target_label
            print(f"  将标签 {source_label} 合并到 {target_label}")

# 直接保存float16，无需额外转换
print(f"\n保存数据到文件（后缀: {OUTPUT_SUFFIX}）...")
np.save(f'X_train_{Sample_time_len}_{OUTPUT_SUFFIX}.npy', X_train)
np.save(f'X_val_{Sample_time_len}_{OUTPUT_SUFFIX}.npy', X_val)
np.save(f'X_test_{Sample_time_len}_{OUTPUT_SUFFIX}.npy', X_test)
np.save(f'y_train_{Sample_time_len}_{OUTPUT_SUFFIX}.npy', y_train)
np.save(f'y_val_{Sample_time_len}_{OUTPUT_SUFFIX}.npy', y_val)
np.save(f'y_test_{Sample_time_len}_{OUTPUT_SUFFIX}.npy', y_test)

print("\n数据保存完成！")
print(f"文件列表:")
print(f"  - X_train_{Sample_time_len}_{OUTPUT_SUFFIX}.npy")
print(f"  - X_val_{Sample_time_len}_{OUTPUT_SUFFIX}.npy")
print(f"  - X_test_{Sample_time_len}_{OUTPUT_SUFFIX}.npy")
print(f"  - y_train_{Sample_time_len}_{OUTPUT_SUFFIX}.npy")
print(f"  - y_val_{Sample_time_len}_{OUTPUT_SUFFIX}.npy")
print(f"  - y_test_{Sample_time_len}_{OUTPUT_SUFFIX}.npy")

# ==========================================
# 可视化检查
# ==========================================
plt.figure(figsize=(10, 6))
plt.pcolor(X_train[0, :, :, 0].astype(np.float32))  # pcolor需float32
plt.colorbar()
plt.title(f'Training Sample 1 (Label: {y_train[0]})')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

print("\n标签分布统计:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_val, counts_val = np.unique(y_val, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print("训练集:", dict(zip(unique_train, counts_train)))
print("验证集:", dict(zip(unique_val, counts_val)))
print("测试集:", dict(zip(unique_test, counts_test)))