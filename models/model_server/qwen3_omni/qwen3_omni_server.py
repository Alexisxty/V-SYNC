import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import argparse
import warnings
import logging
import traceback
from flask import Flask, request, jsonify

from config.settings import CONFIG
from config.paths import PATHS

# GPU配置 - 使用双卡H100（必须在导入transformers前设置）
SPECIFIED_GPUS = CONFIG.model("qwen3_omni").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", []) or [4, 5]  # 两张H100 80GB 显存
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, SPECIFIED_GPUS))

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from models.model_server.local_common.transformers_compat import ensure_qwen3_omni_config_compat

warnings.filterwarnings('ignore')

app = Flask(__name__)
logger = logging.getLogger("qwen3_omni_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    model_log_dir = PATHS.results_logs / "qwen3_omni"
    model_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = model_log_dir / "server.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 全局配置
MODEL_PATH = CONFIG.model("qwen3_omni").get("model_path") or "/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct"
USE_AUDIO_IN_VIDEO = CONFIG.model("qwen3_omni").get("use_audio_in_video", True)
MAX_TOKENS = CONFIG.model("qwen3_omni").get("max_tokens", 50)

# 全局变量
model = None
processor = None
model_loaded = False


def _parse_bool(value, default=True):
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def load_model():
    """加载模型到指定的双GPU"""
    global model, processor, model_loaded

    if model_loaded:
        return

    try:
        # 设置使用指定的两张GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, SPECIFIED_GPUS))

        print(f"Loading model到GPU {SPECIFIED_GPUS}...")

        ensure_qwen3_omni_config_compat()
        # 使用 transformers 加载模型（自动优化，跨双卡分布）
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",  # 自动在多GPU间分配
            dtype='auto'
        )

        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        model_loaded = True
        print(f"Model loaded successfully！Using GPU {SPECIFIED_GPUS}")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        raise


def build_conversation(video_path, question):
    """构建对话格式"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question},
            ],
        }
    ]


def process_video_analysis(video_path, question, use_video, use_audio):
    """处理视频分析"""
    global model, processor
    use_audio_in_video = USE_AUDIO_IN_VIDEO and use_audio

    # 构建对话
    messages = build_conversation(video_path, question)

    # 准备输入
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    if not use_audio:
        audios = None
    if not use_video:
        images = None
        videos = None

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # 推理
    result = model.generate(
        **inputs,
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=MAX_TOKENS,
        thinker_do_sample=False,
        use_audio_in_video=use_audio_in_video,
        return_audio=False
    )

    sequences = None
    if hasattr(result, "sequences"):
        sequences = result.sequences
    elif isinstance(result, tuple) and result:
        sequences = result[0].sequences if hasattr(result[0], "sequences") else result[0]
    elif isinstance(result, str):
        return result
    else:
        sequences = result

    response = processor.batch_decode(
        sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "gpus": SPECIFIED_GPUS
    })


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """分析视频接口 - 只支持文件上传方式"""
    global model, processor

    if not model_loaded:
        return jsonify({"error": "模型未加载"}), 500

    temp_dir = None
    temp_path = None

    try:
        # 只支持文件上传方式
        if 'video' not in request.files:
            return jsonify({"error": "未上传视频文件"}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "未选择文件"}), 400

        question = request.form.get('question', '')
        use_video = _parse_bool(request.form.get("use_video"), True)
        use_audio = _parse_bool(request.form.get("use_audio"), USE_AUDIO_IN_VIDEO)
        if not question.strip():
            return jsonify({"error": "问题不能为空"}), 400

        # 保存上传的文件到临时目录
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(temp_path)

        # 处理视频分析
        answer = process_video_analysis(temp_path, question, use_video, use_audio)

        # 简化的响应格式
        return jsonify({
            "status": "success",
            "answer": answer.strip()
        })

    except Exception as e:
        logger.error("Analyze failed: %s", e)
        logger.error("Traceback:\n%s", traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    finally:
        # 清理临时文件
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Failed to clean up temporary files: {str(e)}")


def parse_args():
    """解析命令行参数"""
    default_host = CONFIG.model("qwen3_omni").get("host") or "127.0.0.1"
    default_port = CONFIG.model("qwen3_omni").get("port") or 5090
    parser = argparse.ArgumentParser(description="Qwen3 Omni Video Analysis Server")
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="服务器端口 (默认: 5090)"
    )
    parser.add_argument(
        "--host",
        default=default_host,
        help="服务器主机地址 (默认: 127.0.0.1)"
    )
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 启动时加载模型
    load_model()

    # 启动服务器
    print(f"Starting server: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
