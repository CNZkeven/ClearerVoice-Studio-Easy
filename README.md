# ClearerVoice-Studio-Easy

ClearerVoice-Studio-Easy 是一个基于深度学习的语音处理工具集，提供多种语音增强、语音分离、目标说话人提取和语音超分辨率功能。本项目是 [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) 的简化版本，专注于核心推理功能。

## 功能特性

### 支持的任务

1. **语音增强 (Speech Enhancement)**
   - `MossFormer2_SE_48K` - 48kHz 语音增强模型
   - `FRCRN_SE_16K` - 16kHz 语音增强模型
   - `MossFormerGAN_SE_16K` - 16kHz GAN 语音增强模型

2. **语音分离 (Speech Separation)**
   - `MossFormer2_SS_16K` - 16kHz 语音分离模型（支持 2 人分离）

3. **语音超分辨率 (Speech Super-Resolution)**
   - `MossFormer2_SR_48K` - 8kHz 到 48kHz 语音超分辨率

4. **目标说话人提取 (Target Speaker Extraction)**
   - `AV_MossFormer2_TSE_16K` - 音视频双模态目标说话人提取

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，用于 GPU 加速)

## 安装步骤

### 方法一：使用 Conda（推荐）

```bash
# 克隆仓库
git clone https://github.com/your-repo/ClearerVoice-Studio-Easy.git
cd ClearerVoice-Studio-Easy

# 创建并激活 conda 环境
conda create -n clearvoice python=3.8
conda activate clearvoice

# 安装依赖
pip install -r requirements.txt
```

### 方法二：使用现有 Conda 环境

```bash
# 激活现有环境
conda activate Common

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 命令行测试

运行测试脚本验证模型加载：

```bash
python test_model.py
```

### 2. Python API 使用

#### 语音增强示例

```python
from clearvoice import ClearVoice

# 加载语音增强模型
myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

# 处理单个音频文件
output_wav = myClearVoice(input_path='samples/input.wav', online_write=False)
myClearVoice.write(output_wav, output_path='samples/output.wav')

# 批量处理音频文件
myClearVoice(input_path='samples/input_folder/', online_write=True, output_path='samples/output_folder/')
```

#### 语音分离示例

```python
from clearvoice import ClearVoice

# 加载语音分离模型
myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

# 处理混合音频
output_wav = myClearVoice(input_path='samples/mixed.wav', online_write=False)
myClearVoice.write(output_wav, output_path='samples/separated/')
```

#### 语音超分辨率示例

```python
from clearvoice import ClearVoice

# 加载语音超分辨率模型
myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

# 处理低分辨率音频
output_wav = myClearVoice(input_path='samples/low_res.wav', online_write=False)
myClearVoice.write(output_wav, output_path='samples/high_res.wav')
```

### 3. Web 界面

启动 Flask Web 服务器：

```bash
cd clearvoice
python app.py
```

访问 `http://localhost:5000` 使用 Web 界面上传音频文件并进行处理。

## 模型下载

模型权重文件会自动从 HuggingFace 下载，存储位置：
- Windows: `checkpoints/模型名称/`
- Linux/Mac: `checkpoints/模型名称/`

手动下载模型：

```python
from huggingface_hub import snapshot_download

# 下载特定模型
snapshot_download(
    repo_id="alibabasglab/MossFormer2_SE_48K",
    local_dir="checkpoints/MossFormer2_SE_48K"
)
```

## 项目结构

```
ClearerVoice-Studio-Easy/
├── clearvoice/
│   ├── __init__.py          # ClearVoice 主类
│   ├── network_wrapper.py   # 网络模型加载器
│   ├── networks.py          # 模型定义
│   ├── models/              # 模型实现
│   │   ├── mossformer2_se/  # 语音增强模型
│   │   ├── mossformer2_ss/  # 语音分离模型
│   │   ├── mossformer2_sr/  # 语音超分辨率模型
│   │   ├── frcrn_se/        # FRCRN 语音增强模型
│   │   └── av_mossformer2_tse/ # 音视频目标说话人提取模型
│   ├── dataloader/          # 数据加载器
│   ├── utils/               # 工具函数
│   ├── config/              # 配置文件
│   │   └── inference/       # 推理配置
│   ├── app.py               # Flask Web 应用
│   └── streamlit_app.py     # Streamlit Web 应用
├── checkpoints/             # 模型权重文件
├── test_model.py            # 测试脚本
├── requirements.txt         # 依赖包列表
└── README.md                # 项目文档
```

## 配置说明

各模型的推理参数配置位于 `clearvoice/config/inference/` 目录下，包含：
- 采样率设置
- 窗口长度和偏移
- FFT 参数
- 解码窗口大小

## 常见问题

### 模型下载失败

如果遇到模型下载失败，可以：
1. 检查网络连接
2. 使用镜像源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
3. 手动从 HuggingFace 下载模型文件

### GPU 内存不足

如果遇到 GPU 内存不足：
1. 减小 `one_time_decode_length` 参数值
2. 使用 CPU 模式：设置 `use_cuda=0`

### 音频采样率不匹配

确保输入音频的采样率与模型要求的采样率一致：
- 48K 模型：需要 48000Hz 音频
- 16K 模型：需要 16000Hz 音频

## 许可证

本项目采用 Apache License 2.0 许可证。

## 致谢

- 原始项目：[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
- 模型来自阿里巴巴达摩院

## 联系方式

如有问题或建议，请在 GitHub Issues 中提交。
