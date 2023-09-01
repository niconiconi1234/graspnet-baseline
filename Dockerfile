FROM alpine/git as git-env
WORKDIR /app
RUN git clone https://github.com/niconiconi1234/graspnet-baseline.git && \
    cd graspnet-baseline && \
    git clone https://github.com/graspnet/graspnetAPI.git


FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
COPY --from=git-env /app /app
WORKDIR /download
# Nvidia has updated the key, so we need to get the new one. For more info, see: https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key del 7fa2af80 && \
    rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get install wget -y && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y
WORKDIR /app/graspnet-baseline
ENV CUDA_HOME=/usr/local/cuda
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && \
    cd pointnet2 && \
    python3 setup.py install && \
    cd ../knn && \
    python3 setup.py install && \
    cd ../graspnetAPI && \
    pip install . 
WORKDIR /app/graspnet-baseline
CMD ["python3", "demo_http_server.py", "--checkpoint_path","checkpoint/checkpoint-rs.tar"]