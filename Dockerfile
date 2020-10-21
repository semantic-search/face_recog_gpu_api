# Install face recognition dependencies
FROM nvidia/cuda:10.0-cudnn7-devel
# Install face recognition dependencies
RUN apt update -y; apt install -y git cmake libsm6 libxext6 libxrender-dev
# Install compilers
RUN apt install -y software-properties-common
RUN apt-get install -y  libopenblas-dev liblapack-dev
# Configure the build for our CUDA configuration.

RUN apt-get install -y python3 python3-pip

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools


RUN apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    python3-dev \
    virtualenv \
    swig

#Install dlib
RUN git clone https://github.com/jainal09/dlib.git /dlib
RUN mkdir -p dlib/build
RUN cmake -H/dlib -B/dlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
RUN cmake --build /dlib/build
RUN cd /dlib; python3 /dlib/setup.py install
#Install Python dependencies
RUN pip3 --no-cache-dir install -r requirements.txt
RUN apt-get install -y libx11-dev libgtk-3-dev
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
ADD . .
ENV LANG C.UTF-8
EXPOSE 8000