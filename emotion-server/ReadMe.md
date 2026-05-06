sudo systemctl restart docker

镜像构建：
docker build -t emotion-server:latest .

镜像加载：
docker load -i emotion-server.tar

sudo docker stop emotion-server
sudo docker rm emotion-server


镜像运行（暴露5000端口）：
docker run -d  --name emotion-server   -p 5000:5000   emotion-server:latest

请求示例：
{
  "filename": "example.jpg",
  "timestamp": 1682851200,
  "image": "<图片的 Base64 字符串>"
}

返回示例：
{
  "filename": "example.jpg",
  "timestamp": 1682851200,
  "emotion": "高兴"
}

服务运行：
python3 server.py

测试脚本运行：
python3 test_request.py -d ./test_dataset/
刚刚运行服务的时候会花费一些时间加载模型，大概需要两三分钟，这段时间内请求不会有返回
pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple
python -m pip install testresources -i https://mirrors.aliyun.com/pypi/simple
python -m pip install --no-cache-dir 