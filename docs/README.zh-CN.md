# V-SYNC（中文说明）

V-SYNC 是用于评估 Omni-LLM 原生联合推理能力的基准测试框架。项目统一了配置、模型服务端入口与评测 pipeline，便于快速对比不同模型与消融设置。

## 快速开始

### 环境准备

- Python >= 3.10
- CUDA 兼容 GPU（本地模型推理需要）
- uv 包管理器

```bash
uv sync
```

### 配置

统一配置位于 `config/config.yaml`，敏感信息通过 `config/.env` 或环境变量注入：

- API Key / Base URL
- 模型温度、top_p、max_tokens
- 模型路径、GPU ID、server_url
- Prompt 模板
- Pipeline 模态（音频/视频消融）

参考 `config/.env.example`。

### 数据准备

```bash
uv run tools/data_tools/01_unzip_data.py
uv run tools/data_tools/02_pipeline.py
```

## 运行方式

### 1) 启动本地模型服务端

每个模型只保留服务端入口（不再使用 unified 脚本）。示例：

```bash
uv run models/model_server/omnivinci/omnivinci_server.py
uv run models/model_server/qwen2_5_omni/qwen_omni_server.py
uv run models/model_server/qwen3_omni/qwen3_omni_server.py
uv run models/model_server/qwen3_omni_thinking/qwen3_omni_thinking_server.py
uv run models/model_server/miniomni_2/miniomni2_server.py
```

### 2) 统一评测入口（推荐）

```bash
uv run run_benchmark.py --model qwen3_omni
```

支持断点续跑（如检测到已有结果会提示 Resume）。

## 消融实验（仅视频 / 仅音频 / 音视一起）

通过 `config/config.yaml` 的 `benchmark.level1.modality` 控制：

- `avt`：音频 + 视频（默认）
- `vt`：仅视频（API 模型不传 ASR；本地模型禁用音频输入）
- `at`：仅音频（API 模型仅传 ASR 文本；本地模型禁用视觉输入）

注意：API 模型仅通过“是否传帧/是否注入 ASR 文本”来模拟模态消融；不处理原始音频。

## 目录结构（顶层）

```
V_SYNC/
├── models/     # 模型适配与服务端
├── data/       # 数据存放
├── tools/      # 基准、分析、实验、数据工具
├── config/     # 配置与环境变量
├── results/    # 输出结果
├── docs/       # 文档
├── pyproject.toml
└── README.md
```

### models/ 子结构

```
models/
├── pipeline/      # 统一 Pipeline 接口
├── model_server/  # 各模型服务端与客户端
└── utils/         # 共享工具
```

## 模型扩展（Pipeline 标准）

`models/pipeline/` 定义统一接口：`InferenceRequest` → `InferenceResult`。新增模型建议：

1. 在 `models/model_server/<model_name>/client.py` 实现客户端；
2. 在 `models/model_server/clients.py` 注册；
3. 用 `run_benchmark.py` 走统一评测。

## 许可证

请根据发布计划补充 LICENSE。
