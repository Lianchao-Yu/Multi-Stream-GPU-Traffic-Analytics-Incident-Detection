# 项目开发 SOP：Multi-Stream GPU Traffic Analytics & Incident Detection

> 版本：v1.0（开发计划，非已实现功能）  
> 开发平台：本地 VS Code + WSL2 Ubuntu（Windows 主机）或原生 Linux；NVIDIA RTX 4060  
> 主要语言：C++17 / CUDA C++；Python 仅用于模型检查或必要的 ONNX 导出  
> 核心库：FFmpeg、OpenCV、ONNX Runtime CUDA Execution Provider、CMake、nlohmann/json  
> 总投入：核心版约 95–130 小时；性能扩展额外 15–25 小时，按实测瓶颈决定  
> 项目性质：基于真实智慧交通业务的**独立研发原型**；不宣称为厦门路桥信息公司的实习交付。

## 0. 业务定义与边界

### 0.1 场景

面向道路/隧道固定摄像机的视频运营辅助系统。系统从**预录制的交通视频**重放多个独立视频源，识别车辆、维持车辆轨迹，生成两种业务输出：

1. **车辆通过计数**（MVP、可定量评价）：按摄像头定义的 ROI / 虚拟计数线 / 运动方向，每辆车只计一次，按类别和时间窗口汇总。
2. **持续静止车辆候选告警**（扩展）：在用户指定的非拥堵监测 ROI 内，车辆保持近似静止超过持续时间后生成事件，并在移动/离开后恢复。初版称为 *stopped-vehicle candidate*，不是已经确认的事故。

实时性目标是**可配置的处理延迟与可控积压**，并不保证 RTX 4060 一定能同时完成 4 路 1080p/30FPS 的逐帧检测。最终处理能力由测试结果决定。

### 0.2 不做的功能

不开发 CNN 架构、不训练全新检测器、不开发 RTSP 编解码协议、不做跨摄像机 Re-ID、不做真实世界车速估计、不做事故类型分类、不做 Web Dashboard、不使用 ROS2/Jetson/实体摄像头、不将后续代码追记为 2025 实习成果。

### 0.3 明确哪些贡献由自己开发

- **第三方**：视频解码使用 FFmpeg；检测网络使用合法取得的预训练 ONNX 模型及 ONNX Runtime GPU 算子；基础图像操作参考 OpenCV；跟踪可用现成算法实现作参考基线。
- **自己完成**：C++ 任务/队列设计、各流时间戳保留、轨迹与事件状态管理、动态 Batch 调度、CUDA 图像预处理 Kernel、GPU Buffer Reuse、性能采样、数据集评价脚本和端到端测试。

## 1. 系统端到端数据流

```
独立录制视频 file1.mp4 / file2.mp4 / ...
       │   每条视频独立 FFmpeg 解码线程
       ▼
DecodedFrame {stream_id, session_id, frame_id, pts_us, bgr}
       │   每路有界帧队列（建议容量 3–5）
       ▼
BatchScheduler (max_batch_size=4, max_wait_ms=10–20 起步)
       │   不同路帧打包，同时保留每帧 stream_id/pts
       ▼
CUDA Preprocess (BGR U8 → letterbox/resize → NCHW FP32)
       │   Batch GPU Tensor
       ▼
ONNX Runtime GPU Vehicle Detector
       │   Detection {class, score, bbox, source_frame_id}
       ▼
Per-Stream Tracker (CPU)
       │   Track {track_id, state, history, pts}
       ▼
Traffic Event Engine (CPU)
       │   计数线穿越 + 持续静止候选事件
       ▼
JSONL Events + Counts CSV + Debug Video + Benchmark Report
```

**时间语义**：事件持续时长和计数发生时刻以**解码视频 PTS**为准；现实播放模式另测 wall-clock 的排队与处理延迟。跳帧时仍保留时间戳差值。不同流不共享 Track ID 命名空间。

## 2. 本地开发环境 P0（预计 5–8h）

### 2.1 推荐环境

1. Windows 驱动已启用 RTX 4060；安装 WSL2 Ubuntu；VS Code 安装 WSL、C/C++、CMake Tools 插件。
2. 项目放在 WSL 的 `~/projects/traffic-gpu-analytics/`，而不是频繁访问 `/mnt/c` 的大批量视频文件目录。
3. 安装 `build-essential cmake ninja-build pkg-config libopencv-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev nlohmann-json3-dev`。
4. 按 NVIDIA WSL CUDA 官方教程安装 **CUDA Toolkit**，不要在 WSL 安装 Linux NVIDIA 显示驱动。
5. 用 `nvidia-smi`、`nvcc --version` 和一个 `cudaGetDeviceProperties()` Smoke Test 确认 GPU 可用；RTX 4060 为 `sm_89`。
6. 为 ONNX Runtime C++ GPU 版本固定兼容的 CUDA/cuDNN/ORT 组合；在 `docs/environment.md` 中记录版本与安装来源，先让独立 ONNX C++ 推理 Smoke Test 成功，再整合 FFmpeg。

示例验证：

```bash
nvidia-smi
nvcc --version
cmake --version
pkg-config --modversion libavformat opencv4
```

**P0 Gate**：CPU-only CMake 工程构建成功；CUDA Smoke Test 成功；至少一个预训练模型在 C++ CPU 或 GPU 端完成单张图片推理。不能只凭 PyTorch 能使用 CUDA 就认为 C++ ORT GPU 已配置好。

## 3. 数据计划 P1（预计 6–10h，可与解码并行）

### 3.1 真实公开数据

- **主任务：AI City Challenge 2021 Track 1**——车辆计数。其视频和规则含 ROI、Movement of Interest；可用提供的参考标注/规则进行计数验证。不要把 Track 1 宣称为停车事件数据集。
- **第二任务：AI City Challenge 2021 Track 4**——交通异常视频。公开异常标注是视频级开始/结束时间，可包括碰撞与抛锚停车；不能直接当作“停止车辆逐车位置或 ID”真值。
- **停车候选单独评估**——从许可允许的视频中挑 10–20 段包含停车或正常拥堵的短片，自己标注 ROI、停车开始/结束时刻、事件有无。公开标注方法与样本数量。

### 3.2 测试数据划分

- 先只下载/处理一个较小子集（如 3–5 个视频），避免一开始下载几十 GB。
- 按**原始视频/摄像头**划分测试集合，避免同一长视频截取的相邻片段同时用于开发调参和最终评分。
- 第三方数据不直接上传仓库，README 提供数据获取来源、授权限制和校验说明。
- 若只拿到约 410p 的视频，检测准确率只报告该原始分辨率。将其上采样至 1080p 可作为**系统压力测试**，不能称为真实 1080p 视觉精度验证。

**P1 Gate**：至少两个不同场景的原始视频可解码；PTS、分辨率、FPS、视频长度记录正确；至少有一个计数 ROI 配置文件。

## 4. 分阶段开发与验收

### Phase 1：单路视频解码和回放（预计 10–14h）

**实现文件**：`src/media/video_reader.cpp`、`src/media/frame_converter.cpp`、`src/app/main.cpp`。

**功能**：
- FFmpeg `avformat_open_input` / `avformat_find_stream_info` 获取视频流。
- `avcodec_send_packet` + `avcodec_receive_frame` 解码，`sws_scale` 规范为 BGR8（或保留 NV12 的后续扩展接口）。
- 从 decoded frame 的最佳可用时间戳转换为 `pts_us`，不要直接把 `frame_index / 30` 假定成真实时间。
- `--mode offline`：不等待，测最大处理吞吐量；`--mode paced`：按视频 PTS 模拟实时播放。
- 视频结束、坏帧、解码异常有显式状态；返回的像素内存具有明确生命周期。

**验收**：单路视频逐帧读取与 OpenCV/FFmpeg 可视化时间轴基本一致；1000 帧测试无崩溃；进度统计输出 JSON；终止后无未释放解码资源。

### Phase 2：单路 CNN 检测 Baseline（预计 12–18h）

**实现文件**：`src/detection/onnx_detector.cpp`、`src/detection/postprocess.cpp`、`src/cpu/preprocess.cpp`。

**功能**：
- 选一款许可适合公开项目的**现成预训练车辆检测 ONNX 模型**，冻结模型参数。确定 `input shape / BGR→RGB / mean,std / letterbox / bbox output / NMS`，不能假设所有 YOLO 导出的 ONNX 输出格式一样。
- CPU OpenCV Resize / Letterbox / HWC→CHW FP32。
- ONNX Runtime 先以 CPU 跑通，再启用 CUDA EP；记录实际使用的 Execution Provider。
- 将模型输出框从模型输入坐标逆映射回原始视频帧坐标。
- 可选：使用绘制 BBox 的调试视频证明输出，不将可视化视为准确率评估。

**验收**：同一批 20 张图的 CPU/GPU 检测结果近似一致（浮点数容差 + 框的 IoU）；输出带 class、score、frame ID 与原图 bbox 的 JSONL；无 GPU 运行时静默回退到 CPU 的未记录情况。

### Phase 3：多目标跟踪、车辆计数及停车候选事件（预计 18–24h）

**实现文件**：`src/tracking/tracker.cpp`、`src/events/counting.cpp`、`src/events/stopped_vehicle.cpp`、`src/events/event_writer.cpp`。

**功能**：
- 初版可采用一个可解释的 CPU Tracker：Kalman 预测 + IoU 数据关联 + track 生命期（Tentative / Confirmed / Lost / Removed），或使用依法许可的 ByteTrack 实现做参考。
- 不同 stream 独立 tracker，绝对不能将其他视频中的车辆合并为同一个 ID。
- 车辆计数：按 ROI/方向/虚拟线规则触发，`counted_track_ids` 防重复；如使用 AI City 官方计数数据，须对齐其 ROI、MOI 和计数时点定义，而不是任意画一条线。
- 停车候选：使用 bbox 底部中心点的历史相对位移 + 持续时间（PTS）；在固定摄像头、指定非信号灯/非交叉路口 ROI 中启用。
- 状态机：`MOVING → STOP_CANDIDATE → STOP_CONFIRMED → RECOVERED`，加入最短持续时间、丢帧容忍窗口、恢复阈值、防重复事件。
- 初版不要把像素速度当作真实 m/s；拥堵、红灯、遮挡和摄像头抖动会导致误判，必须列为失败案例。

**验收**：合成的 5 种轨迹单元测试全部通过（正常过线、方向相反、短暂遮挡、停车后恢复、重复检测）；真实视频的车辆计数误差可计算；停车样本的事件区间可与人工标注比较。

### Phase 4：多路流管理和动态 Batch 调度（预计 15–20h）

**实现文件**：`src/runtime/stream_worker.cpp`、`src/runtime/bounded_queue.hpp`、`src/runtime/batch_scheduler.cpp`、`src/runtime/pipeline.cpp`。

**功能**：
- 1 路 → 2 路 → 4 路渐进增加独立录制视频。
- 每路 3–5 帧的有界队列，慢处理时选择 `drop_oldest` 或 `block` 策略；记录 dropped frames。
- 调度器支持 `max_batch_size` 与 `max_wait_ms`，例如 4 和 15ms 为**初始实验值**，最终以结果为准。
- 给 batch 中每张帧附加 `{stream_id, session_id, frame_id, pts_us, input_size}`，推理完后正确路由回所属 Tracker。
- 单路数据不足时可以发送 Partial Batch，但要让推理模型和运行时后端支持相应 batch shape。
- 断流/EOF 后不会阻塞其他流；视频重新播放时重置 session，避免历史 track 混入。

**验收**：不同 FPS、不同分辨率和不同视频长度混合输入仍正确；0/1/2/4 路配置正常退出；队列长度不超上限；batch 不乱流、不丢 ID；每路计数与单路运行相比误差在预先定义的容差内。

### Phase 5：自己开发 CUDA Preprocessing & GPU 内存优化（预计 15–22h）

**实现文件**：`src/cuda/preprocess_kernels.cu`、`src/cuda/gpu_workspace.cu`、`src/detection/gpu_detector.cpp`。

**功能**：
- 实现并行的 BGR U8 → RGB / Letterbox / Resize / Normalize / NCHW FP32 的图像到张量 Kernel；使用从原图到网络输入的 transform metadata 还原框。
- 支持 batch 输入和不同原图大小，但模型推理张量为固定或约束范围内的形状。
- RAII GPU Buffer，减少每帧 `cudaMalloc/cudaFree`；先单 Stream 正确实现，再测是否需要异步 H2D、Pinned Memory、多 Stream。
- 使用 ORT CUDA EP；条件允许时采用 **I/O Binding** 让 GPU 输入直接进入推理图，避免 GPU→CPU→GPU 不必要往返。
- 记录 `cudaEvent` 的 GPU Kernel / H2D 时间；不能使用 GPU Kernel 时间冒充完整响应延迟。

**验收**：CPU 和 CUDA 输入张量在定义容差内一致，Letterbox 边界区域正确；Batch=1/2/4 输出车辆框坐标正确；没有未释放显存；相比 CPU OpenCV + GPU 推理 Baseline 的性能差异可重复测量。

### Phase 6：事件准确率、鲁棒性与系统性能（预计 14–20h）

**实现文件**：`benchmarks/run_benchmarks.py`、`tests/test_event_state.cpp`、`tests/test_batch_scheduler.cpp`、`tests/test_geometry.cpp`、`docs/report.md`。

**三组系统对照**：
- A：CPU Preprocess + CPU Inference（功能与结果参考）。
- B：CPU Preprocess + GPU Inference（量化现成 GPU 推理收益）。
- C：**自写 CUDA Preprocess + GPU Inference + Dynamic Batching**（量化自主实现的增量价值）。

**性能维度**：
- Offline：总输出帧数 / 总 wall-clock 时间（FPS）、视频小时/小时、GPU 利用率和显存峰值。
- Paced：p50/p95 input-to-result latency、Batch 等待时间、视频落后播放进度的 Lag、每路丢帧率。
- 分阶段：decode、queue wait、H2D、preprocess、inference、tracking、event emission。
- Accuracy：车辆计数绝对误差/MAE；停止候选事件 Precision/Recall、误报数量、确认延迟（限**自己已标注的子集**）。
- 不把 AI City Track 4 的视频级异常区间解释为逐车停车标注，也不把普通拥堵自动定义为事故。

**故障注入**：坏视频、空视频、某路提前 EOF、突发帧速率、一个输入源断开、模型加载失败、CUDA 显存申请失败（模拟时可降低可用显存上限）。

**验收**：完成至少 1/2/4 路和 2 种模型推理批次的表格；包含真实失败案例；项目可从干净的 Git checkout 按 README 构建并复现实验。

### Phase 7（可选）：更深度 GPU Systems 优化（额外 15–25h）

仅在 Phase 6 profiling 发现明确瓶颈后实施：

- GPU Tensor IoBinding、异步传输重叠、固定内存池、不同 batch 配置比较。
- 若 CUDA 预处理已明显不是瓶颈，不再继续增加 kernel fusion；改为分析排队与解码瓶颈。
- 可用 TensorRT 作为第二推理后端，但**不需要**自己写卷积算子；注意模型兼容、引擎和许可证。
- 对比「静态 batch」与「deadline-aware batch」吞吐量/尾延迟，以实验结论决定复杂策略是否值得。

## 5. 建议仓库目录（文件名可按开发实际精简）

```text
traffic-gpu-analytics/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── LICENSE
├── configs/
│   ├── single_stream.json
│   ├── multi_stream.json
│   └── camera_rois.json
├── include/traffic/
│   ├── types.hpp
│   ├── video_reader.hpp
│   ├── detector.hpp
│   ├── tracker.hpp
│   ├── batch_scheduler.hpp
│   ├── event_engine.hpp
│   ├── gpu_preprocessor.hpp
│   └── metrics.hpp
├── src/
│   ├── app/main.cpp
│   ├── media/{video_reader.cpp,frame_converter.cpp}
│   ├── detection/{onnx_detector.cpp,postprocess.cpp}
│   ├── tracking/tracker.cpp
│   ├── events/{counting.cpp,stopped_vehicle.cpp,event_writer.cpp}
│   ├── runtime/{stream_worker.cpp,batch_scheduler.cpp,pipeline.cpp}
│   ├── cpu/preprocess.cpp
│   └── cuda/{preprocess_kernels.cu,gpu_workspace.cu}
├── tests/
│   ├── test_tracker.cpp
│   ├── test_counting.cpp
│   ├── test_stopped_vehicle.cpp
│   ├── test_batch_scheduler.cpp
│   └── test_preprocess.cpp
├── benchmarks/
│   ├── run_benchmarks.py
│   └── analyze_results.py
├── models/README.md
├── data/README.md
├── results/.gitkeep
└── docs/{architecture.md,environment.md,evaluation.md,report.md}
```

## 6. C++ 模块接口草案（说明性的设计接口，非直接可编译实现）

```cpp
struct Frame {
    int stream_id;
    uint64_t session_id;
    uint64_t frame_id;
    int64_t pts_us;
    cv::Mat bgr;          // owning or ref-counted; queue lifetime must be safe
};

struct Detection {
    cv::Rect2f bbox;     // source-frame pixel coords
    int class_id;
    float confidence;
};

struct Track {
    int stream_id;
    uint64_t track_id;
    cv::Rect2f bbox;
    int64_t last_pts_us;
    int age_frames;
    bool confirmed;
};

struct InferenceFrameResult {
    int stream_id;
    uint64_t frame_id;
    int64_t pts_us;
    std::vector<Detection> detections;
};

class IVideoReader {
public:
    virtual ~IVideoReader() = default;
    virtual bool read(Frame& out) = 0; // false on EOS; errors via Status separately
};

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual std::vector<InferenceFrameResult> inferBatch(
        const std::vector<Frame>& batch) = 0;
};

class ITracker {
public:
    virtual ~ITracker() = default;
    virtual std::vector<Track> update(
        const std::vector<Detection>& detections,
        int64_t pts_us) = 0;
};

struct TrafficEvent {
    int stream_id;
    uint64_t track_id;
    std::string event_type;    // "vehicle_counted", "stopped_candidate", ...
    int64_t timestamp_us;
    std::string roi_id;
    float confidence;
};

class ITrafficEventEngine {
public:
    virtual ~ITrafficEventEngine() = default;
    virtual std::vector<TrafficEvent> update(
        int stream_id, int64_t pts_us,
        const std::vector<Track>& tracks) = 0;
};

struct BatchPolicy {
    int max_batch_size = 4;
    int max_wait_ms = 15;
    int queue_capacity = 4;
};

class IBatchScheduler {
public:
    virtual ~IBatchScheduler() = default;
    virtual std::vector<Frame> nextBatch() = 0;
};
```

**重要接口约束**：真实实现应额外使用 `Status/Result` 表示错误或 EOF，避免把 `bool false` 同时表示二者；GPU 异步工作流需在缓冲区可安全复用之前完成相应同步/事件等待。`std::vector<Frame>` 携带较大的 `cv::Mat` 时要理解 OpenCV 引用计数，防止跨线程共享可变底层像素数据。

## 7. 关键配置及命令示例

`configs/multi_stream.json` 示例：

```json
{
  "sources": [
    {"id": 0, "path": "data/cam_001.mp4", "roi_config": "cam_001"},
    {"id": 1, "path": "data/cam_002.mp4", "roi_config": "cam_002"}
  ],
  "mode": "paced",
  "backend": "onnx_cuda",
  "gpu_preprocess": true,
  "max_batch_size": 4,
  "max_wait_ms": 15,
  "queue_capacity": 4,
  "drop_policy": "drop_oldest",
  "detector_model": "models/vehicle_detector.onnx",
  "event_output": "results/events.jsonl",
  "metrics_output": "results/metrics.json"
}
```

目标 CLI（实现完相应代码后）：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j

./build/traffic_analytics --config configs/single_stream.json
./build/traffic_analytics --config configs/multi_stream.json
ctest --test-dir build --output-on-failure
```

> 说明：上述 CMake 和 CLI 是**目标接口**，不是目前已经可运行的程序；不同 ORT 安装方式可能需要额外的 `ONNXRUNTIME_ROOT` 路径和链接配置。

## 8. 测试验收规范（不可省略）

| 里程碑 | 必须保留的证据 | 完成标准 |
|---|---|---|
| P0 | `env.txt` / GPU Smoke Test | C++ CUDA 与 ORT 可用 |
| P1 | `decode_report.json` | PTS 正确、坏帧处理、无异常退出 |
| P2 | `detections.jsonl` | GPU 真实运行、坐标映射正确 |
| P3 | `events.jsonl` / `counts.csv` | 每 Track 仅计数一次，停车事件有状态恢复 |
| P4 | `scheduler_metrics.json` | 多路稳定，缓冲区有上界，不串流 |
| P5 | `preprocess_diff.csv` | CPU/CUDA 数值一致且有测量 |
| P6 | `report.md` / 失败样本 | 完整吞吐量、延迟、业务准确率和限制 |

**建议优先关注的异常测试**：
1. 同一辆车短暂遮挡，不能凭一次漏检重复计数。
2. 车完全离开 ROI 后计数，不因 Bounding Box 在边界抖动反复触发。
3. 红灯/拥堵造成群体静止时，不应自动当作单车故障；若没有可靠抑制规则则明确列为误报限制。
4. 视频有 B 帧或可变帧率时，事件时间仍应基于 PTS 正确计算。
5. 多路其中一路结束时，其他路不能卡死。
6. Batch 的不同帧输出必须回送至正确 stream、track、timestamp。

## 9. 性能报告模板

每组实验记录：输入视频分辨率/真实 FPS、stream 数、模型文件哈希、模型输入尺寸、CPU/GPU 后端、batch size/wait、跑几次、GPU/Driver/CUDA/ORT 版本、是否 warm-up。

### 必要对照表

| 配置 | 路数 | 数据帧率 | 总吞吐 FPS | P95 结果延迟(ms) | 丢帧率 | VRAM 峰值 | 计数 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Preproc + CPU Infer | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| CPU Preproc + GPU Infer | 1/2/4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| CUDA Preproc + GPU Infer | 1/2/4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| CUDA + Dynamic Batch | 1/2/4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

**严格区分**：
- GPU Kernel time ≠ CNN inference time ≠ End-to-end frame latency。
- 离线不按真实播放节奏运行的 throughput ≠ 实时服务的延迟。
- 如果输入或输出经过 JPEG/视频重新编码，必须纳入端到端测量。
- CPU/GPU 推理结果可能有浮点或 NMS 差异，以合理容差比较。

## 10. 止损与范围管理

- **Checkpoint A（P2 后）**：C++ GPU 检测模型无法稳定运行，优先解决 ORT/CUDA 版本与 I/O，而不是继续开发追踪器。
- **Checkpoint B（P3 后）**：车辆计数结果不可靠，先修复时间戳、跟踪和 ROI；暂停多路和 CUDA 细节优化。
- **Checkpoint C（P4 后）**：1–2 路已满足业务要求时，不强行将 4 路 1080p/30FPS 作为最低交付目标。
- **Checkpoint D（P6 后）**：若自定义 CUDA 预处理在完整流水线中无明显收益，保留优化研究及负面结论，不捏造提升。

**核心交付标准**：能把真实交通监控录像变成带时间戳、可解释、可复核的车辆计数及候选事件；同时证明自己设计的 C++ 多路调度/CUDA 预处理在某些负载下的成本与性能取舍。

## 11. 参考资料与许可检查

- AI City Challenge dataset access: https://www.aicitychallenge.org/ai-city-challenge-dataset-access/
- AI City Challenge 2021 counting rules: https://www.aicitychallenge.org/2021-data-and-evaluation/
- AI City Challenge 2021 anomaly time labels: https://www.aicitychallenge.org/2021-data-and-evaluation/
- ONNX Runtime GPU I/O Binding: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
- ONNX Runtime CUDA compatibility: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- NVIDIA WSL CUDA setup: https://docs.nvidia.com/cuda/wsl-user-guide/
- NVIDIA TensorRT batching concepts: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html
- ByteTrack (tracking reference, license to verify): https://github.com/FoundationVision/ByteTrack

在公开 GitHub 仓库前核对：数据集视频分发限制、预训练模型权重许可证、第三方 tracker 或 YOLO 代码许可证、课程 Starter Code 公开要求。仓库只公开自己可合法发布的源代码、测试数据或复现脚本。
