# 开发环境说明（Plan B）

- **日期**：2026-11-09
- **状态**：环境搭建完成，CUDA / FFmpeg+OpenCV / ONNX Runtime CUDA EP / CMake+Ninja 四项 Smoke Test 全部通过
- **范围**：仅环境搭建与验证，未开始业务代码

## 1. 主机与系统

| 项目 | 值 |
|---|---|
| 系统 | Ubuntu 26.04 LTS (Resolute Raccoon), WSL2 |
| 内核 | 6.18.54.1-microsoft-standard-WSL2 |
| CPU | 20 vCPU |
| 内存 | 7.6 GiB（当前可用约 5.6 GiB），Swap 2.0 GiB |
| 磁盘 | 1007 GiB，已用 36 GiB，可用 920 GiB（ext4 /dev/sdd） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| 计算能力 | sm_89（8.9） |
| Windows NVIDIA 驱动 | 621.14（nvidia-smi CUDA UMD 13.5） |
| WSL 内 Linux 驱动 | 未安装，使用 Windows 驱动透传 |

## 2. 软件版本

| 组件 | 版本 | 安装位置 / 来源 |
|---|---|---|
| CUDA Toolkit | 13.5.37（release 13.5, V13.5.37） | `/usr/local/cuda-13.5`，NVIDIA CUDA apt 源 |
| cuDNN | 9.28.1.4（MAJOR 9, MINOR 28, PATCH 1） | `cudnn9-cuda-13-5` apt 包 |
| ONNX Runtime | 1.31.1（CUDA 13 GPU 版） | `~/opt/onnxruntime-1.31.1-cuda13` |
| GCC / G++ | 15.2.0（Ubuntu 15.2.0-16ubuntu1） | 系统 |
| CMake | 4.2.3 | 系统 |
| Ninja | 1.13.2 | `ninja-build` |
| pkg-config | 2.5.1 | 系统 |
| FFmpeg | 8.0.1-3ubuntu2 | 系统 |
| libavformat | 62.3.100 | `libavformat-dev` |
| libavcodec | 62.11.100 | `libavcodec-dev` |
| libavutil | 60.8.100 | `libavutil-dev` |
| libswscale | 9.1.100 | `libswscale-dev` |
| OpenCV | 4.10.0 | `libopencv-dev` |
| nlohmann/json | 头文件 3.11.3（包版本 3.12.0.really.3.12.0.really.3.11.3-3build1） | `nlohmann-json3-dev` |
| Python | 3.14.4 | 系统 |
| Python venv | onnx 1.23.2 / numpy 2.5.4 | 项目内 `.venv`，仅用于 ONNX 模型生成/检查 |

> 未安装 `onnxruntime-gpu` Python 包；C++ 项目直接使用 ORT C++ 包。
> Ubuntu 自带的 CUDA 12.4（`/usr/bin/nvcc` 包装脚本，`/usr/lib/nvidia-cuda-toolkit`）保留未删除，通过 PATH / `CMAKE_CUDA_COMPILER` 显式选择 13.5。

## 3. 安装来源

### 3.1 基础依赖

```bash
sudo apt install -y \
  ninja-build pkg-config ffmpeg \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libopencv-dev nlohmann-json3-dev \
  python3.14-venv python3-pip
```

### 3.2 CUDA / cuDNN

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-5 cudnn9-cuda-13-5
```

- 未安装 `cuda` / `cuda-drivers` 元包
- 未在 WSL 内安装 Linux NVIDIA 驱动

### 3.3 ONNX Runtime 1.31.1 CUDA13

```text
URL:    https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-gpu_cuda13-1.31.1.tgz
SHA256: b2146862e691f7e338f5aff9c7a9a5567c2ca57cf01442a1adb3a13bf62d9095
路径:   /home/jason/opt/onnxruntime-1.31.1-cuda13
```

包含 `onnxruntime_cxx_api.h`、`libonnxruntime.so`、`libonnxruntime_providers_cuda.so`、`libonnxruntime_providers_shared.so`。

## 4. 环境变量

已幂等写入 `~/.bashrc`：

```bash
export CUDA_HOME=/usr/local/cuda-13.5
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export ONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-1.31.1-cuda13"
export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH"
```

新开交互终端后：

```bash
which nvcc          # /usr/local/cuda-13.5/bin/nvcc
nvcc --version      # release 13.5, V13.5.37
```

## 5. Smoke Test 记录

测试临时目录：`/tmp/opencode/planB_smoke/`（未污染项目根目录）

### 5.1 CUDA Smoke Test

```bash
/usr/local/cuda-13.5/bin/nvcc -std=c++17 -arch=sm_89 smoke.cu -o smoke
./smoke
```

结果：

```text
devices=1
GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU sm_89 mem=8.00 GiB
kernel result sample=6.0 all_ok=yes
```

验证点：`cudaGetDeviceCount`、`cudaGetDeviceProperties`、简单 kernel、C++17 标准库（`<vector>`/`<algorithm>`/`<memory>`/`<string>`）与 GCC 15 协同正常。

### 5.2 FFmpeg / OpenCV / nlohmann JSON 链接测试

```bash
g++ -std=c++17 -O2 ffmpeg_opencv_json.cpp \
  $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale opencv4) \
  -o ffmpeg_opencv_json
./ffmpeg_opencv_json
```

结果：

```text
avformat_version=4064100 (8.0.1-3ubuntu2)
avcodec_version=4066148
OpenCV=4.10.0
json={"project":"gpu-traffic-analytics","smoke":true,"streams":4}
```

### 5.3 ONNX Runtime CUDA EP 测试

生成 tiny Conv 模型（输入 `1x3x32x32`，Conv 3→8，3x3，输出 `1x8x32x32`）：

```bash
.venv/bin/python make_tiny_onnx.py   # -> /tmp/opencode/planB_smoke/tiny.onnx
```

编译与运行：

```bash
g++ -std=c++17 -O2 -I"$ONNXRUNTIME_ROOT/include" ort_cuda_test.cpp \
  -L"$ONNXRUNTIME_ROOT/lib" -lonnxruntime \
  -Wl,-rpath,"$ONNXRUNTIME_ROOT/lib" -o ort_cuda_test

export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
./ort_cuda_test
```

结果：

```text
available providers:
  - TensorrtExecutionProvider
  - CUDAExecutionProvider
  - CPUExecutionProvider
input=input output=output
output shape: 1 8 32 32
output[0..4]: -0.272981 -0.158066 -0.263589 -0.450291 -0.479071
ORT CUDA EP run OK
```

- `Ort::GetAvailableProviders()` 包含 `CUDAExecutionProvider`
- Session 创建并成功运行，输出维度正确
- 日志中无 `not assigned` / `fallback` / `failed` / `error` 告警（verbose 日志）
- 动态库依赖检查：`libonnxruntime_providers_cuda.so` 与 `libonnxruntime.so` 均无 `not found`

### 5.4 CMake + Ninja + CUDA + ORT + OpenCV 工具链测试

临时工程：`/tmp/opencode/planB_smoke/cmake_test/`

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.5 \
  -DONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-1.31.1-cuda13"

cmake --build build -j
./build/planb_smoke
```

结果：

```text
OpenCV=4.10.0
FFmpeg avformat=4064100 avcodec=4066148
ORT providers: TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider
```

## 6. 已知限制与 workaround

1. **WSL 内不安装 Linux NVIDIA 驱动**：GPU 由 Windows 驱动透传，`nvidia-smi` 显示 Driver 621.14 / CUDA UMD 13.5。
2. **RTX 4060 Laptop 8 GB 显存**：后续多路 1080p + batch 推理需实测显存上限，不能假设 4 路 1080p/30FPS 必然达标。
3. **WSL 内存约 7.6 GiB**：多路解码 + 队列 + ORT 需控制帧缓冲和队列容量。
4. **CMake 可能优先选中 Ubuntu CUDA 12.4**：`/usr/bin/nvcc` 是包装脚本，指向 `/usr/lib/nvidia-cuda-toolkit`。仅设置 PATH 时 CMake 仍可能选中它，导致 `ptxas fatal: Value 'sm_52' is not defined`（CUDA 13 已移除 sm_52）。必须显式指定：
   ```bash
   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc
   ```
5. **CMake 4.2 + OpenCV 4.10 兼容**：本次临时工程使用了 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`，用于规避旧 `cmake_minimum_required` 策略错误。正式 CMakeLists 若未出现该错误可不加。
6. **Ubuntu apt 源速度问题**：`archive.ubuntu.com` 在此网络下只有约 15 kB/s，已切换为 `https://mirrors.kernel.org/ubuntu/`（实测约 8.6 MB/s）。`/etc/apt/sources.list.d/ubuntu.sources` 已修改，并保留 `*.bak-*` 备份。
7. **CUDA 12.4 与 13.5 共存**：未卸载系统 CUDA 12.4 包；所有构建显式使用 `/usr/local/cuda-13.5`。
8. **Python venv 仅用于模型处理**：`.venv` 中只有 `numpy` / `onnx`，不安装 `onnxruntime-gpu`；C++ 推理使用独立 ORT C++ 包。

## 7. 下一步建议

1. **P0 工程骨架**：在项目根目录建立正式 `CMakeLists.txt` / `CMakePresets.json`，固定：
   - C++17、`CMAKE_CUDA_ARCHITECTURES=89`
   - `CUDAToolkit_ROOT=/usr/local/cuda-13.5`
   - `CMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc`
   - `ONNXRUNTIME_ROOT=~/opt/onnxruntime-1.31.1-cuda13`
2. **Phase 1 解码**：单路 FFmpeg 解码 + `sws_scale` → BGR，保留 PTS，先做 1000 帧稳定性测试。
3. **Phase 2 检测**：先 CPU ORT 跑通，再切 CUDA EP；记录实际 provider，禁止静默 CPU fallback。
4. 后续如需升级驱动/CUDA/cuDNN/ORT，必须保持 CUDA 13 / cuDNN 9 / ORT CUDA13 三者匹配，并更新本文档。

## 8. 验收清单

- [x] `nvidia-smi` Driver 621.14（>= 580），CUDA UMD 13.5
- [x] `nvcc --version` 为 13.5（`/usr/local/cuda-13.5/bin/nvcc`）
- [x] CUDA Smoke Test 通过，识别 sm_89
- [x] FFmpeg / OpenCV / nlohmann 头文件与链接测试通过
- [x] `Ort::GetAvailableProviders()` 包含 `CUDAExecutionProvider`
- [x] tiny ONNX 模型在 CUDA EP 上运行成功
- [x] 无未记录的 CPU fallback
- [x] CMake + Ninja + CUDA + ORT + OpenCV 临时工程构建并运行成功
- [x] `docs/environment.md` 已生成
