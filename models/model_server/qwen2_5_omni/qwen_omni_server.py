import torch
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import argparse
from flask import Flask, request, jsonify
import logging
from config.paths import PATHS
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from config.settings import CONFIG

app = Flask(__name__)
logger = logging.getLogger("qwen2_5_omni_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    model_log_dir = PATHS.results_logs / "qwen2_5_omni"
    model_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = model_log_dir / "server.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 简化的系统提示词
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# 全局配置
MODEL_PATH = CONFIG.model("qwen2_5_omni").get("model_path") or "/publicssd/xty/models/Qwen2.5-Omni-7B"
USE_AUDIO_IN_VIDEO = CONFIG.model("qwen2_5_omni").get("use_audio_in_video", True)
MAX_TOKENS = CONFIG.model("qwen2_5_omni").get("max_tokens", 50)
TEMPERATURE = CONFIG.model("qwen2_5_omni").get("temperature", 0.1)

# GPU配置 - 支持环境变量和命令行参数
def get_gpu_id():
    """获取GPU编号，优先级：命令行参数 > 环境变量 > 默认值0"""
    # 首先检查环境变量
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu_id is not None:
        try:
            return int(gpu_id)
        except ValueError:
            pass
    model_gpus = CONFIG.model("qwen2_5_omni").get("gpu_ids", [])
    if model_gpus:
        return int(model_gpus[0])
    runtime_gpus = CONFIG.runtime("gpu_ids", [])
    if runtime_gpus:
        return int(runtime_gpus[0])
    return 5

# 全局变量
gpu_id = get_gpu_id()
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
    """加载模型到指定GPU"""
    global model, processor, model_loaded, gpu_id

    if model_loaded:
        return

    try:
        print(f"Loading model到GPU {gpu_id}...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=f"cuda:{gpu_id}",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        model_loaded = True
        print(f"Model loaded successfully！Using GPU {gpu_id}")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        raise


def build_conversation(video_path, question):
    """构建对话格式"""
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question},
            ],
        },
    ]


def process_video_analysis(video_path, question, use_video, use_audio):
    """处理视频分析"""
    global model, processor
    use_audio_in_video = USE_AUDIO_IN_VIDEO and use_audio
    
    # 构建对话
    conversation = build_conversation(video_path, question)
    
    # 准备输入
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    if not use_audio:
        audios = None
    if not use_video:
        images = None
        videos = None
    logger.info("Request media: video=%s audio=%s", use_video, use_audio)
    
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

    # 推理 - 添加max_tokens限制
    text_ids, _ = model.generate(
        **inputs, 
        use_audio_in_video=use_audio_in_video,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True
    )
    result = processor.batch_decode(
        text_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    return result[0] if result else ""


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok", 
        "model_loaded": model_loaded,
        "gpu_id": gpu_id
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
    default_host = CONFIG.model("qwen2_5_omni").get("host") or "127.0.0.1"
    default_port = CONFIG.model("qwen2_5_omni").get("port") or 5089
    parser = argparse.ArgumentParser(description="Qwen Omni Video Analysis Server")
    parser.add_argument(
        "--gpu-id", 
        type=int, 
        default=None,
        help="指定GPU编号 (默认: 0，或通过CUDA_VISIBLE_DEVICES环境变量指定)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=default_port,
        help="服务器端口 (默认: 5089)"
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
    
    # 更新GPU编号（如果通过命令行指定）
    if args.gpu_id is not None:
        gpu_id = args.gpu_id
        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"通过命令行参数设置Using GPU {gpu_id}")
    else:
        print(f"Using GPU {gpu_id} (通过环境变量或默认值)")
    
    # 启动时加载模型
    load_model()
    
    # 启动服务器
    print(f"Starting server: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
