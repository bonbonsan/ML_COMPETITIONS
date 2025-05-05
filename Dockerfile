# 1. CUDA & Python 3.11 ベースイメージ
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 2. 基本パッケージ
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils \
    curl git && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 3. Python3.11 をデフォルトに
RUN rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3


# 4. 作業ディレクトリを作成（Paperspace上でもこのパスになる）
WORKDIR /workspace

# 5. pipパッケージのインストール（事前にrequirements.txtをコピー）
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 6. 残りのソースコード一式をコピー（my_library や projectsなど）
COPY . .

# 7. PYTHONPATHを通す（my_library を import 可能にする）
ENV PYTHONPATH=/workspace

# 8. 起動時にbashへ（対話的に使える）
CMD ["/bin/bash"]
