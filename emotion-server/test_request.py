# test_client.py

import os
import time
import base64
import argparse
import requests

def iter_images(dir_path):
    """遍历目录，读取支持的图片文件并返回 (filename, timestamp, b64str)"""
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        full = os.path.join(dir_path, fn)
        if not os.path.isfile(full):
            continue
        # 读取二进制并转 base64（去掉换行）
        with open(full, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        # 时间戳可以用文件修改时间或当前时间
        ts = int(os.path.getmtime(full))
        yield fn, ts, b64

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dir', required=True,
        help='待测试图片所在目录'
    )
    parser.add_argument(
        '-u', '--url', default='http://localhost:5000/predict',
        help='情绪识别服务地址'
    )
    args = parser.parse_args()

    for fn, ts, b64 in iter_images(args.dir):
        payload = {
            "filename": fn,
            "timestamp": ts,
            "image": b64
        }
        try:
            resp = requests.post(args.url, json=payload, timeout=1000)
            print(f"{fn} -> {resp.status_code} {resp.json()}")
        except Exception as e:
            print(f"{fn} -> ERROR: {e}")

if __name__ == "__main__":
    main()
