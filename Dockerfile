FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y vim \
        # python3.9 \
        python3-pip \
        python-is-python3
RUN apt-get update && apt-get install -y git wget unzip \
        pbzip2 pv bzip2 cabextract iputils-ping pdsh coinor-cbc tmux

# Update the env with the CUDA
ENV PATH="usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' >> /root/.bashrc

# Install dependencies
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 numpy==1.24.1 xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121
RUN pip install packaging==24.0

WORKDIR /workspace/

# Install apex
RUN pip install --upgrade pip setuptools
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout bae1f93d033716dc9115a0baf7bcda328addabe9 && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Install requirements for Megatron
RUN apt-get install -y cmake
RUN pip install nvidia-cudnn-cu12==8.9.7.29
ENV CUDNN_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/
RUN echo 'export CUDNN_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/' >> /root/.bashrc
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@release_v1.3
RUN pip install pybind11

# Install requirements
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .
RUN cd csrc/fused_dense_lib && \
    pip install . && \
    cd ../layer_norm && \
    pip install .

# # OPENMPI
# ENV OPENMPI_BASEVERSION=4.1
# ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.0
# RUN mkdir -p /build && \
#     cd /build && \
#     wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
#     cd openmpi-${OPENMPI_VERSION} && \
#     ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
#     make -j"$(nproc)" install && \
#     ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
#     # Sanity check:
#     test -f /usr/local/mpi/bin/mpic++ && \
#     cd ~ && \
#     rm -rf /build

# # Create a wrapper for OpenMPI to allow running as root by default
# RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
#     echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
#     echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
#     chmod a+x /usr/local/mpi/bin/mpirun

# RUN echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /root/.bashrc && \
#     echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /root/.bashrc

# # Needs to be in docker PATH if compiling other items & bashrc PATH (later)
# ENV PATH=/usr/local/mpi/bin:${PATH} \
#     LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}

LABEL version="1.0"
LABEL description="Mist dockerfile"
