
# [Automated 3D reconstruction from satellite images](https://www.siam-is18.dm.unibo.it/minitutorials)
### SIAM IS18 Mini-tutorial - 08/06/2018


<img src="docs/step1_150.png" >
<img src="docs/step2_150.png" >
<img src="docs/step3_150.png" >
<img src="docs/step4_150.png" >



### Abstract

Commercial spaceborne imaging is experiencing an unprecedented growth both in size of the constellations and resolution of the images. This is driven by applications ranging from geographic mapping to measuring glacier evolution, or rescue assistance for natural disasters. For all these applications it is critical to automatically extract and update elevation data from arbitrary collections of multi-date satellite images. This multi-date satellite stereo problem is a challenging application of 3D computer vision: images are taken at very different dates, from very different points of view, and under different lighting conditions. The case of urban scenes adds further difficulties because of occlusions and reflections. 

This tutorial is a hands-on introduction to the manipulation of optical satellite images, using complete examples with python code. The objective is to provide all the tools needed to process and exploit the images for 3D reconstruction. We will present the essential modeling elements needed for building a stereo pipeline for satellite images. This includes the specifics of satellite imaging such as pushbroom sensor modeling, coordinate systems, and localization functions. Then we will review the main concepts and algorithms for stereovision and tailor them to the case of satellite images. Finally, we will bring together these elements to build a 3D reconstruction pipeline for multi-date satellite images.


* [Slides](docs/slidesIS18.pdf)
* [Notebook sources](https://github.com/gfacciol/IS18notebook)

### See also

* [Satellite Stereo Pipeline (S2P)](https://github.com/MISS3D/s2p)
* [Automatic 3D Reconstruction from Multi-Date Satellite Images (winning solution of the IARPA Multi-View Stereo 3D Mapping Challenge 2016)](https://gfacciol.github.io/multi-date-stereo/)
* [Master2 MVA course: Large-scale geospatial data exploitation](https://mvaisat.wp.imt.fr/)


### LIVE servers running the notebook

* [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/gfacciol/IS18notebook/master?filepath=IS18.ipynb)
* <a href="http://menthe.ovh.hw.ipol.im:8080/">server 1</a>
* <a href="http://avocat.ovh.hw.ipol.im:8080/">server 2</a>
* <a href="http://piadina.ovh.hw.ipol.im:8080/">server 3</a>
* <a href="http://tigella.ovh.hw.ipol.im:8080/">server 4</a>



### Run the notebook with a Docker image on http://localhost:8000

**requires a good internet connection to access the satellite images** 


1. Download and launch [the docker image](https://hub.docker.com/r/facciolo/is18notebook) with:

       # Type ctrl-D to exit the container
       docker run --rm \
           -p 8000:8000 \
           -t -i  facciolo/is18notebook \ 
           jupyter notebook --port=8000 --ip=0.0.0.0 --NotebookApp.token=''

2. Connect to:  http://localhost:8000




<!-- 
You can run the server locally using a Docker image.
We tested it on Linux and MacOS systems.

* [https://hub.docker.com/r/facciolo/is18-satellite-minitutorial/](https://hub.docker.com/r/facciolo/is18-satellite-minitutorial/)


### To launch the docker image:

1. First create the shared home directory in the host computer:

       mkdir ~/is18tutorial
       # makes writable by the docker use
       chmod o+rwx ~/is18tutorial

2. Then launch the docker instance by calling (automatically downloads it):

       # Type ctrl-D to exit the container
       docker run --rm \
           -v ~/is18tutorial:/home/student1/IS18 \
           -p 8000:8000 -p 8008:8008  \
           --env EXTERNAL_HTTP_SRV_URL=http://localhost:8008 \
           -t -i  facciolo/is18-satellite-minitutorial \
           bash /singleuser_initscript.sh

3. Connect to:    http://localhost:8000
-->   
