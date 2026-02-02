# V-SYNC: Perceiving When to Speak, A Diagnostic Benchmark for Omni Models

![V-SYNC Hero](assets/hero.svg)

> 一个诊断型基准，追问 Omni 模型最核心的问题：**它究竟能否正确把握“何时该说”？**

V-SYNC 评估的是**音视频同步的时间对齐能力**，而不是简单的多模态拼接。任务设计保证**单模态捷径失效**，只有真正统一感知与推理的 Omni 模型才能稳定作答。

本项目基于 AV-SyncBench/omni_benchmark 规范重构，用于发布与工程化落地，因此**评测数据集与结果完全共用**，同时在配置、服务端与评测入口上实现统一。

---

## 目录导航

- [为什么是 V-SYNC](#为什么是-v-sync)
- [设计理念](#设计理念)
- [结果概览](#结果概览与原始项目共用)
- [系统概览](#系统概览)
- [快速开始](#快速开始)
- [启动本地模型服务端](#启动本地模型服务端)
- [统一评测入口（Level1）](#统一评测入口level1)
- [消融实验](#消融实验仅视频--仅音频--音视联合)
- [配置映射表](#配置映射表)
- [目录结构](#目录结构顶层)
- [Level2 / Level3](#level2--level3)
- [API 配置](#api-配置)
- [FAQ](#faq)
- [Citation](#citation)

---

## 为什么是 V-SYNC

多数多模态基准关注“模型懂什么”。V-SYNC 专注于“模型知道**何时**懂”：

- **时间绑定**：能否把正确的说话者与正确时刻对齐
- **跨模态依赖**：任务设计保证单模态不可解
- **原生融合**：避免后融合技巧，要求统一感知与推理

---

## 设计理念

V-SYNC 是**诊断型**而非“刷榜型”基准：

- **简洁任务表面**：用直观问题直接探测同步感知
- **单模态不可解**：在设计上封死捷径
- **时间精度约束**：每个样例绑定明确时间窗
- **评测管线化**：服务端优先、配置驱动、断点续跑

---

## 结果概览（与原始项目共用）

本项目与 AV-SyncBench/omni_benchmark 使用相同评测配置与数据集，结果完全共用：

| Model | Accuracy |
|-------|----------|
| Qwen3-Omni | 77.6% |
| GPT-4o | 33.0% |
| Gemini 2.5 Flash | 21.0% |

---

## 系统概览

```
[Dataset + ASR] -> [Pipeline] -> [Model Client] -> [HTTP Server] -> [Omni Model]
                         |                 |
                         |                 +-- 本地服务端 (Qwen/OmniVinci/MiniOmni2)
                         +-- API 客户端 (GPT-4o / Gemini)
```

- **服务端优先**：本地模型统一走 HTTP
- **统一管线**：本地/远程评测流程一致
- **模态控制**：audio/video 开关集中管理

---

## 快速开始

### 环境准备

- Python >= 3.10
- CUDA 兼容 GPU（本地模型推理需要）
- uv 包管理器

```bash
uv sync
```

### 配置

统一配置位于 `config/config.yaml`，敏感信息通过 `config/.env` 或环境变量注入。

可配置项包括：
- API Key / Base URL
- 模型温度、top_p、max_tokens
- 模型路径、GPU ID、server_url
- Prompt 模板
- 消融模态（audio/video）

---

## 启动本地模型服务端

每个模型仅保留服务端入口（不使用 unified 脚本）：

```bash
uv run models/model_server/omnivinci/omnivinci_server.py
uv run models/model_server/qwen2_5_omni/qwen_omni_server.py
uv run models/model_server/qwen3_omni/qwen3_omni_server.py
uv run models/model_server/qwen3_omni_thinking/qwen3_omni_thinking_server.py
uv run models/model_server/miniomni_2/miniomni2_server.py
```

---

## 统一评测入口（Level1）

```bash
uv run run_benchmark.py --model qwen3_omni
```

若输出文件存在会提示 Resume，支持断点继续。

---

## 消融实验（仅视频 / 仅音频 / 音视联合）

通过 `config/config.yaml` 的 `benchmark.level1.modality` 控制：

- `avt`：音频 + 视频（默认）
- `vt`：仅视频
- `at`：仅音频

说明：
- API 模型通过“是否传帧 / 是否注入 ASR 文本”模拟消融
- 本地模型会禁用对应输入通道

---

## 配置映射表

| 区域 | Key | 示例 |
|------|-----|------|
| API | `api.openai.base_url` | `https://.../v1` |
| API | `api.openai.api_key` | 从 `.env` 读取 |
| Model | `models.<name>.model_path` | `/publicssd/...` |
| Model | `models.<name>.gpu_ids` | `[6,7]` |
| Model | `models.<name>.server_url` | `http://127.0.0.1:5091` |
| Runtime | `runtime.max_retries` | `5` |
| Runtime | `runtime.frame_interval_sec` | `1` |
| Pipeline | `benchmark.level1.modality` | `avt / vt / at` |

---

## 目录结构（顶层）

```
V_SYNC/
├── models/            # model server / client / utils
├── data/              # dataset
├── tools/             # analysis / experiments / data tools (git ignored)
├── config/            # config + env
├── results/           # outputs (local)
├── docs/              # docs
├── run_benchmark.py   # Level1
├── run_benchmark_level2.py
├── run_benchmark_level3.py
└── README.md
```

---

## Level2 / Level3

已准备好独立入口：

```bash
uv run run_benchmark_level2.py --model qwen3_omni
uv run run_benchmark_level3.py --model qwen3_omni
```

当你实现 `models/pipeline/level2_pipeline.py` / `level3_pipeline.py` 并在 `models/pipeline/__init__.py` 导出后即可直接使用。

---

## API 配置

```
OPENAI_API_BASE=...
OPENAI_API_KEY=...
```

OpenAI 兼容客户端会从 `.env` 或环境变量读取。

---

## FAQ

**Q: 一定需要音频+视频吗？**
A: 是的，基准设计保证单模态不可解，但你可以用消融开关做对比实验。

**Q: 为什么必须走服务端？**
A: 保证本地与 API 模型评测路径一致，避免本地私有逻辑干扰。

**Q: 可以断点续跑吗？**
A: 可以，Runner 会检测结果文件并提示 Resume。

---

## Citation

```bibtex
@misc{vsync2025,
  title={V-SYNC: Perceiving When to Speak, A Diagnostic Benchmark for Omni Models},
  author={Alexisxty},
  year={2025},
  url={https://github.com/Alexisxty/V-SYNC}
}
```
