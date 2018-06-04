
The server configuration runs a jupyterhub with a `dockerspawner` spawner 
that launches Docker containers with the image `facciolo/is18-satellite-minitutorial`.
The jupyterhub entrypoint is on port 8000.

A particularity of the setup is the need to display the point clouds produced in the notebook.
An http server must be used to serve the point clouds on port 8008.
For that, all content placed in the `/shared` directory (inside the virtual machine) 
is assumed to be available on the address `EXTERNAL_HTTP_SRV_URL`.

In the server setup the `/shared` mapped to a shared directory on the host computer, 
and `launch.sh` runs an http-server on port 8008.
The details of the volume map, and the variable transfer `EXTERNAL_HTTP_SRV_URL`
are in the file `jupyterhub_config.py`.

----------------------------------------------------------------------------------------

The same Docker image can also be used to run a single user server. 
In that case the script `singleuser_initscript.sh` is provided. 
It runs `jupyter notebook` on port 8000 and launches an `http-server` 
on port 8008 serving the content of `/shared`

```
       docker run --rm \
           -p 8000:8000 -p 8008:8008  \
           --env EXTERNAL_HTTP_SRV_URL=http://localhost:8008 \
           -t -i  facciolo/is18-satellite-minitutorial \
           bash /singleuser_initscript.sh
```
