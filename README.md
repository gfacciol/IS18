# [Automated 3D reconstruction from satellite images](https://gfacciol.github.io/IS18/)

[Website](https://gfacciol.github.io/IS18/) and notebook for the [SIAM IS18 Mini-tutorial - 08/06/2018](https://www.siam-is18.dm.unibo.it/minitutorials)

See https://gfacciol.github.io/IS18/ for details.

Try the notebook here [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/gfacciol/25f186142e0346c8578ab3ca2e5bc696/is18-tutorial-in-colab.ipynb)  and here [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/gfacciol/IS18/master?filepath=IS18.ipynb)  (currently failing)



### Technical info

* The MyBinder image is build using Docker
* The notebook dependences are in requirements.txt 
* This notebook works only with notebook==5.4.1. With newer versions the potree visualization fails to load 

### Updates

* 01/08/2022: Binder image is currently not working, probably an issue with the interaction of jupyter and the requirements. More recent requirements are: 
   `numpy matplotlib scipy geojson pyproj==2.4.1 opencv-contrib-python==4.5.3.56 ad rasterio srtm4 folium rpcm numba pypotree` and they work correctly in colab.
