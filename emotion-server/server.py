# app.py

import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace

# 不使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# DeepFace 分析并映射成中文标签
ZH_MAP = {
    'neutral': '平静',
    'happy':   '高兴',
    'surprise':'惊讶',
    'sad':     '伤心',
    'angry':   '愤怒',
    'disgust': '厌恶',
    'fear':    '恐惧'
}

def analyze_emotion(img: np.ndarray) -> str:
    """
    调用 DeepFace 分析人脸情绪，返回中文标签
    """
    # 注意 enforce_detection=True 会在人脸检测失败时抛异常
    res = DeepFace.analyze(
        img,
        actions=['emotion'],
        detector_backend='mtcnn',
        enforce_detection=True
    )
    label_en = res[0]['dominant_emotion']
    return ZH_MAP.get(label_en, '平静')


@app.route('/predict', methods=['POST'])
def predict():
    """
    接收 JSON：
      {
        "filename": "...",
        "timestamp": 1234567890,
        "image": "....base64..."
      }
    返回 JSON：
      {
        "filename": "...",
        "timestamp": 1234567890,
        "emotion": "高兴"
      }
    """
    try:
        data = request.get_json(force=True)
        img_b64 = data.get('image')
        if not img_b64:
            return jsonify({'error': 'no image provided'}), 400

        # Base64 解码到 OpenCV 图像
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("decode image fail")

        # 分析表情
        label_zh = analyze_emotion(img)

        # 返回结果
        return jsonify({
            'filename': data.get('filename'),
            'timestamp': data.get('timestamp'),
            'emotion': label_zh
        })

    except Exception as e:
        # 捕获 DeepFace 检测不到人脸等异常
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 可改 host/port 或部署到 gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)
