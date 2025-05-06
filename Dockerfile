# 1. CUDA & Python 3.11 ベースイメージ
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 2. 基本パッケージ
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils \
    curl git && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 3. Python3.11 をデフォルトに
RUN rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# 4. 作業ディレクトリを作成（Paperspace上でもこのパスになる）
WORKDIR /workspace

# 5. .env.example を .env にコピー（.env は .dockerignore によって無視されている前提）
COPY .env.example .env

# 6. requirements.txt をコピーして install（GPU対応修正済み）
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# 7. Jupyter起動スクリプトをコピーして実行可能にする
COPY scripts/jupyter-lab.sh /usr/local/bin/jupyter-lab.sh
RUN chmod +x /usr/local/bin/jupyter-lab.sh

# 8. 残りのソースコード（my_libraryやprojectsなど）をコピー
COPY . .

# 9. PYTHONPATH を通す（自作ライブラリの import のため）
ENV PYTHONPATH=/workspace

# 10. コンテナ起動時は bash（VSCodeターミナルなどで使う前提）
CMD ["/bin/bash"]
