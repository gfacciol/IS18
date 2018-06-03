# docker for the MVA course

FROM ubuntu:xenial
MAINTAINER Gabriele Facciolo <gfacciol@gmail.com>
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-add-repository -y ppa:ubuntugis/ppa
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
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
    nodejs-legacy \
    python3 \
    python3-gdal \
    python3-pip \
    python3-numpy-dev \
    software-properties-common \
    imagemagick \
    vim \
    cmake \
    unzip \
    g++-7
RUN add-apt-repository ppa:certbot/certbot -y

# https stuff: TODO manually in interactive mode
# apt-get update & apt-get install certbot
# certbot certonly --register-unsafely-without-email

## MISPLACED CV2 DEPENDENCY
RUN apt update && apt install -y libsm6 libxext6

RUN pip3 install -U pip

# install TSD requirements
RUN git clone https://github.com/carlodef/tsd.git --branch master --single-branch --depth 1 --recursive
RUN pip3 install -r tsd/requirements.txt
RUN pip3 install enum34 features tifffile tifffile filelock

RUN pip3 install matplotlib scipy \
                 pyproj pyfftw \
                 scikit-image \
                 numba pillow \
                 opencv-contrib-python \
                 ad

# create user accounts
#RUN useradd -m -p $(openssl passwd -1 ed188d37503aa2fe3cd65bc51516c41a) carlo
RUN useradd -m -p $(openssl passwd -1 password) student1



# install jupyterhub/jupyterlab
RUN pip3 install jupyterhub-systemdspawner
RUN pip3 install jupyter jupyterhub notebook

RUN npm install n -g   &&   n stable
RUN npm install -g configurable-http-proxy


# install widget stuff
RUN pip3 install \
    pandas \
    folium \
    ipywidgets \
    ipyleaflet 

#RUN pip install jupyterlab==0.32.0rc1
#RUN jupyter labextension install jupyter-leaflet
### https://github.com/jupyter-widgets/ipywidgets/tree/master/packages/jupyterlab-manager#version-compatibility
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.33
RUN    jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
       jupyter nbextension enable --py --sys-prefix ipyleaflet
#RUN jupyter labextension install jupyterlab-toc


## more in https://github.com/topics/jupyterlab-extension


### jupyterhub extensions
RUN pip3 install jupyter_contrib_nbextensions && \
   jupyter contrib nbextension install --system && \
   jupyter nbextension enable --sys-prefix scratchpad/main 
#&& \
#    jupyter nbextension enable --sys-prefix varInspector/main && \
#    jupyter nbextension enable --sys-prefix collapsible_headings/main && \
#    jupyter nbextension enable --sys-prefix toc2/main


## notebook
RUN cd /home/ && wget menthe.ovh.hw.ipol.im:8080/x.tgz  &&  tar xzvf x.tgz
RUN cd /home/ && chown -R student1.student1 student1
RUN cd /home/student1 && make 


## export jupyter to pdf
#RUN apt-get update && apt-get install -y \
#    pandoc \
#    texlive-xetex 

# hack
RUN npm install -g http-server

# create a user, since we don't want to run as root
ENV HOME=/home/student1
WORKDIR $HOME
EXPOSE 8000:8000
EXPOSE 8001:8001
#USER student1
#
#CMD ["jupyterhub-singleuser"]
#CMD jupyter notebook --port=8000 --ip=* --allow-root --NotebookApp.token=''
# runs jupyter on 8000 and http-server on 8001
COPY initscript.sh initscript.sh
CMD ./initscript.sh

