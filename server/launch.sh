cd /home/facciolo/IS18/docker_servers
# mount shared from another server
#sshfs -o allow_other,nonempty avocat.ovh.hw.ipol.im:/home/facciolo/IS18/docker_servers/shared shared

jupyterhub &
cd shared && http-server -p 8008

