# ClearerVoice-Studio-Easy

一个面向推理落地的语音处理项目，提供简化版的 **ClearVoice** 工作流，支持语音增强、语音分离、目标说话人提取和语音超分辨率，并额外集成了可选的 **Rust 原生扩展** 用于部分音频预处理加速。


<img width="1892" height="1338" alt="image" src="https://github.com/user-attachments/assets/c4dfb33f-e044-4e80-9a6e-f483b8ce318b" />


本仓库适合以下场景：

- 快速体验 ClearVoice 推理能力
- 在本地搭建一个可运行的 Web 演示
- 通过 Python API 集成到自己的工具链或服务中
- 在 Windows 环境下使用已编译好的 `clearvoice_native` 原生扩展

## Features

- 支持多种语音任务：增强、分离、目标说话人提取、超分辨率
- 提供 Python API，适合脚本化处理和二次开发
- 提供 Flask Web 界面，便于本地演示
- 已加入 `clearvoice_native` Rust 扩展，用于加速部分音频处理逻辑
- 项目结构相对精简，便于阅读、调试和继续开发

## Supported Tasks & Models

### 1. Speech Enhancement

- `MossFormer2_SE_48K`
- `FRCRN_SE_16K`
- `MossFormerGAN_SE_16K`

### 2. Speech Separation

- `MossFormer2_SS_16K`

### 3. Speech Super-Resolution

- `MossFormer2_SR_48K`

### 4. Target Speaker Extraction

- `AV_MossFormer2_TSE_16K`

## Project Structure

```text
ClearerVoice-Studio-Easy/
├─ clearvoice/                 # Python 主程序与 Web 界面
│  ├─ clearvoice/              # 核心推理代码
│  ├─ templates/               # Flask 页面模板
│  ├─ app.py                   # Flask Web 入口
│  ├─ streamlit_app.py         # Streamlit 演示入口（可选）
│  └─ pyproject.toml           # Python 包定义
├─ clearvoice_native/          # Rust 原生扩展（PyO3 + maturin）
│  ├─ src/
│  ├─ Cargo.toml
│  ├─ pyproject.toml
│  └─ clearvoice_native.cp313-win_amd64.pyd
├─ docs/plans/                 # 本项目内的设计与实施记录
├─ requirements.txt            # 根目录依赖清单
├─ build_rust.bat              # Windows 下 Rust 扩展构建脚本
├─ test_model.py               # 模型加载检查脚本
└─ test_process.py             # 处理流程检查脚本
```

## Environment Requirements

### 基础要求

- Python `>= 3.8`
- `pip` 或 `conda`
- 建议使用独立虚拟环境

### 推荐运行环境

- Windows 10 / 11
- Python 3.10 ~ 3.13
- NVIDIA GPU + CUDA 版 PyTorch（可选，但推荐）

### Rust 扩展额外要求

如果你需要重新构建 `clearvoice_native`：

- Rust toolchain
- Visual Studio C++ Build Tools 或 Visual Studio 2022
- Windows SDK
- `maturin`

> 当前仓库已经包含一份 Windows 下可用的 `.pyd` 构建结果，但如果你的 Python 版本、平台或 ABI 不一致，仍然建议重新构建。

## Installation

下面给出两种推荐安装方式。

### 方式一：直接安装项目依赖

```bash
git clone https://github.com/<your-name>/ClearerVoice-Studio-Easy.git
cd ClearerVoice-Studio-Easy

python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e ./clearvoice
```

### 方式二：使用 Conda 环境

```bash
git clone https://github.com/<your-name>/ClearerVoice-Studio-Easy.git
cd ClearerVoice-Studio-Easy

conda create -n clearvoice python=3.13 -y
conda activate clearvoice

pip install --upgrade pip
pip install -r requirements.txt
pip install -e ./clearvoice
```

## Install PyTorch

请根据你的硬件环境单独安装 PyTorch。

### CUDA 版本示例

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### CPU 版本示例

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> 如果你已经通过 `requirements.txt` 安装过 PyTorch，可跳过这一步；但在很多机器上，按官方源重新安装更稳妥。

## Model Checkpoints

模型权重默认从 Hugging Face 下载，建议存放在仓库根目录下的 `checkpoints/`：

```text
checkpoints/
├─ MossFormer2_SE_48K/
├─ FRCRN_SE_16K/
├─ MossFormerGAN_SE_16K/
├─ MossFormer2_SS_16K/
├─ MossFormer2_SR_48K/
└─ AV_MossFormer2_TSE_16K/
```

如需手动下载，可参考：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="alibabasglab/MossFormer2_SE_48K",
    local_dir="checkpoints/MossFormer2_SE_48K"
)
```

## Quick Start

### 1. Python API

在脚本中使用 `ClearVoice`：

```python
from clearvoice import ClearVoice

engine = ClearVoice(
    task='speech_enhancement',
    model_names=['MossFormer2_SE_48K']
)

result = engine(input_path='samples/input.wav', online_write=False)
engine.write(result, output_path='samples/output.wav')
```

### 2. 批量处理目录

```python
from clearvoice import ClearVoice

engine = ClearVoice(
    task='speech_enhancement',
    model_names=['MossFormer2_SE_48K']
)

engine(
    input_path='samples/input_folder',
    online_write=True,
    output_path='samples/output_folder'
)
```

### 3. 语音分离示例

```python
from clearvoice import ClearVoice

engine = ClearVoice(
    task='speech_separation',
    model_names=['MossFormer2_SS_16K']
)

result = engine(input_path='samples/mixed.wav', online_write=False)
engine.write(result, output_path='samples/separated')
```

### 4. 超分辨率示例

```python
from clearvoice import ClearVoice

engine = ClearVoice(
    task='speech_super_resolution',
    model_names=['MossFormer2_SR_48K']
)

result = engine(input_path='samples/low_res.wav', online_write=False)
engine.write(result, output_path='samples/high_res.wav')
```

## Run Web Demo

项目内提供了一个基于 Flask 的本地 Web 界面：

```bash
cd clearvoice
python app.py
```

启动后访问：

```text
http://127.0.0.1:5000
```

功能包括：

- 选择任务与模型
- 上传音频文件
- 本地执行推理
- 下载处理结果

## Optional: Run Streamlit Demo

仓库里还包含 `clearvoice/streamlit_app.py`，适合做快速交互演示。

```bash
cd clearvoice
streamlit run streamlit_app.py
```

> 注意：该入口依赖 `silero_vad` 等额外组件，如果你只需要稳定运行，优先使用 Flask 版本。

## Rust Native Extension

`clearvoice_native` 是一个基于 **PyO3 + maturin** 的 Rust 扩展，当前用于加速部分音频处理函数，例如：

- `audio_norm`
- `overlap_add_segments`
- `bandwidth_sub`

Python 侧已经实现了 **优先调用 Rust、失败时自动回退到 Python 实现** 的机制，因此即使扩展不可用，主流程仍可运行。

### 在当前环境中验证扩展

```bash
python -c "from clearvoice_native import audio_norm, overlap_add_segments, bandwidth_sub; print('OK')"
```

### Windows 下重新构建扩展

```bash
cd clearvoice_native
python -m pip install maturin
python -m maturin build --release
python -m pip install --force-reinstall dist\clearvoice_native-0.1.0-cp313-cp313-win_amd64.whl
```

## Basic Verification

### 检查模型是否能正常加载

```bash
python test_model.py
```

### 检查处理流程接口是否可调用

```bash
python test_process.py
```

### 检查 `ClearVoice` 是否可导入

```bash
cd clearvoice
python -c "from clearvoice import ClearVoice; print('Import OK')"
```

## Troubleshooting

### 1. 模型无法加载

请优先检查：

- `checkpoints/` 下是否存在对应模型目录
- 是否已成功下载权重文件
- Python 环境中的依赖是否完整

### 2. Web 页面可以打开，但处理失败

通常与以下问题有关：

- 模型权重未准备好
- 输入音频格式异常
- 当前环境中的 PyTorch / torchaudio 版本不兼容

### 3. Rust 扩展编译失败

Windows 下常见原因：

- 未安装 Visual Studio C++ 工具链
- 未安装 Windows SDK
- `maturin` 未安装
- 当前 Python 版本与已有 `.pyd` 不匹配

### 4. `import clearvoice_native` 导入到的是仓库目录而不是编译结果

本仓库已经处理了这个问题：

- 安装 wheel 后可从 `site-packages` 导入
- 仓库目录下也保留了 `__init__.py` 与 `.pyd` 入口，便于本地开发

## Acknowledgements

本项目基于以下工作进行简化与整理：

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
- PyTorch
- Hugging Face Hub
- PyO3 / maturin

## License

本仓库当前使用的许可证文件见：`LICENSE`

