# docker for the MVA course

FROM ubuntu:gutsy
MAINTAINER Gabriele Facciolo <gfacciol@gmail.com>
RUN apt-add-repository -y ppa:ubuntugis/ppa
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN add-apt-repository -y ppa:deadsnakes/ppa

#RUN apt-get update && apt-get install -y python3.6
#RUN apt-get update && apt-get install -y software-properties-common

RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    geographiclib-tools \
    git \
    libfftw3-dev \
    libgdal-dev \
    libgeographic-dev \
    libgeotiff-dev \
    libtiff5-dev \
    npm \
    python3 \
    python3-gdal \
    python3-pip \
    python3-numpy-dev \
    software-properties-common \
    imagemagick \
    vim \
    cmake \
    unzip \
    g++-7 \ 
    curl

#    nodejs-legacy \

## MISPLACED CV2 DEPENDENCY
RUN apt update && apt install -y libsm6 libxext6


## export jupyter to pdf
#RUN apt-get update && apt-get install -y \
#    pandoc \
#    texlive-xetex 




# CREATE USER 
ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}


# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}


#RUN pip3 install -U pip
#RUN python3 -m pip install --user --upgrade pip 
#RUN curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py &&  python3 get-pip.py --force-reinstall


# install requirements
RUN pip3 install -r ${HOME}/requirements.txt



RUN    jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
       jupyter nbextension enable --py --sys-prefix ipyleaflet


### jupyterhub extensions
RUN pip3 install jupyter_contrib_nbextensions && \
   jupyter contrib nbextension install --system && \
   jupyter nbextension enable --sys-prefix varInspector/main && \
   jupyter nbextension enable --sys-prefix scratchpad/main 
#&& \
#    jupyter nbextension enable --sys-prefix collapsible_headings/main && \
#    jupyter nbextension enable --sys-prefix toc2/main



# override notebook with a version that works with potree
RUN pip3 install notebook==5.4.1 tornado==5.1.1



# NODE 
#RUN npm install n -g   &&   n stable
#RUN npm install -g configurable-http-proxy
## single user server hack: run a http-server on the /shared directory
#RUN npm install -g http-server




# compile SRTM
#RUN cd ${HOME}/srtm4 && make

# compile potreeconverter
RUN cd /home/ && git clone https://github.com/gfacciol/PotreeConverter_PLY_toolchain.git && cd /home/PotreeConverter_PLY_toolchain && git submodule update --init --recursive && CC=gcc-7 CXX=g++-7 make && cp -r /home/PotreeConverter_PLY_toolchain/PotreeConverter/PotreeConverter/resources /home/PotreeConverter_PLY_toolchain/PotreeConverter/build/PotreeConverter/

# switch to user
USER ${NB_USER}
WORKDIR $HOME

# create a user, since we don't want to run as root
EXPOSE 8000:8000
#EXPOSE 8008:8008

#
#CMD ["jupyterhub-singleuser"]
CMD jupyter notebook --port=8000 --ip=0.0.0.0

