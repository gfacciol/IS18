# [https://www.siam-is18.dm.unibo.it/minitutorials Automated 3D reconstruction from satellite images]


### Abstract

    Commercial spaceborne imaging is experiencing an unprecedented growth both in size of the constellations and resolution of the images. This is driven by applications ranging from geographic mapping to measuring glacier evolution, or rescue assistance for natural disasters. For all these applications it is critical to automatically extract and update elevation data from arbitrary collections of multi-date satellite images. This multi-date satellite stereo problem is a challenging application of 3D computer vision: images are taken at very different dates, from very different points of view, and under different lighting conditions. The case of urban scenes adds further difficulties because of occlusions and reflections. 
    This tutorial is a hands-on introduction to the manipulation of optical satellite images, using complete examples with python code. The objective is to provide all the tools needed to process and exploit the images for 3D reconstruction. We will present the essential modeling elements needed for building a stereo pipeline for satellite images. This includes the specifics of satellite imaging such as pushbroom sensor modeling, coordinate systems, and localization functions. Then we will review the main concepts and algorithms for stereovision and tailor them to the case of satellite images. Finally, we will bring together these elements to build a 3D reconstruction pipeline for multi-date satellite images.


## LIVE servers

* <a href="https://menthe.ovh.hw.ipol.im:8000/">server 1</a>
* <a href="https://avocat.ovh.hw.ipol.im:8000/">server 2</a>



## Docker image running on http://localhost:8000

You can run the server locally using Docker.
We tested this on Linux and MacOS systems.

First create the shared home directory in the host computer:

    mkdir ~/IS18tutorial
    # makes writable by the docker use
    chmod o+rwx ~/IS18tutorial

Then run the docker instance by calling (automatically downloads it):

    # Type ctrl-D to exit the container
    docker run --rm  \
        -v ~/IS18tutorial:/home/student1  \
        -p 8000:8000  --workdir /home/student  -t -i  facciolo/IS18-satellite-minitutorial  \
        jupyter notebook --port=8000 --ip=* --allow-root --NotebookApp.token=''

Connect to:    http://localhost:8000
