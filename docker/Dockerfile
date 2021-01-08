FROM cupy/cupy

# Install basic packages
RUN apt-get update -y && apt-get upgrade -y && \
        apt-get install -y wget git libpython3-dev build-essential && \
        apt-get autoremove -y && apt-get clean -y && apt-get autoclean -y

# Install python3 modules
RUN pip3 install --upgrade --no-cache-dir numpy matplotlib Pillow

# Install pytorch
RUN pip3 install torch torchvision

RUN cd /opt && git clone https://github.com/sniklaus/pytorch-liteflownet
WORKDIR /opt/pytorch-liteflownet
