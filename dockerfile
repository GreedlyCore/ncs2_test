FROM openvino/ubuntu20_runtime:2022.3.1
ENV DEBIAN_FRONTEND=noninteractive
ENV QT_DEBUG_PLUGINS=1
ENV MYRIAD_NO_BOOT=1       

USER root 
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    wget \
    iputils-ping \
    usbutils \
    curl \
    net-tools \
    apt-transport-https \
    ca-certificates \
    nano \
    sudo \
    udev \
    lsb-release \
    build-essential \
    cmake \
    git \
    && rm -r /var/lib/apt/lists/*

RUN source /opt/intel/openvino_2022.3.1.9227/setupvars.sh
RUN cd /opt/intel/openvino_2022.3.1.9227/install_dependencies && sudo bash install_openvino_dependencies.sh
RUN cd /opt/intel/openvino_2022.3.1.9227/install_dependencies && sudo bash install_NCS_udev_rules.sh
RUN cd /opt/intel/openvino_2022.3.1.9227/install_dependencies && sudo usermod -a -G users "$(whoami)"
RUN cd /opt/intel/openvino_2022.3.1.9227/install_dependencies && sudo cp 97-myriad-usbboot.rules /etc/udev/rules.d/

# RUN python3 -m venv openvino_env

COPY r.txt r.txt
RUN python3 -m  pip install --upgrade pip
RUN pip install -r r.txt

# RUN source openvino_env/bin/activate

# RUN sudo reboot
# return back to workspace
# RUN cd ~ && cd workspace
# COPY argus argus
# COPY setup.py setup.py
# RUN pip install -e .
