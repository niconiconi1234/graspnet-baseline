FROM alpine/git as git-env
WORKDIR /app
RUN git clone https://github.com/niconiconi1234/graspnet-baseline.git && \
    cd graspnet-baseline && \
    git clone https://github.com/graspnet/graspnetAPI.git


FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
COPY --from=git-env /app /app
WORKDIR /app/graspnet-baseline
ENV CUDA_HOME=/usr/local/cuda
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  --ignore-installed -r requirements.txt && \
    cd pointnet2 && \
    python3 setup.py install && \
    cd ../knn && \
    python3 setup.py install && \
    cd ../graspnetAPI && \
    pip install .
    WORKDIR /app/graspnet-baseline
CMD ["python3", "main.py", "--checkpoint_path","checkpoint/checkpoint-rs.tar"]