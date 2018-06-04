#sudo jupyterhub --ssl-key ./server/conf/live/menthe.ovh.hw.ipol.im/privkey.pem --ssl-cert ./server/conf/live/menthe.ovh.hw.ipol.im/fullchain.pem &
#cd shared
#sudo http-server -K ../server/conf/live/menthe.ovh.hw.ipol.im/privkey.pem -C ../server/conf/live/menthe.ovh.hw.ipol.im/fullchain.pem

cd /home/facciolo/IS18/docker_servers
# mount shared from another server
#sshfs -o allow_other,nonempty avocat.ovh.hw.ipol.im:/home/facciolo/IS18/docker_servers/shared shared

jupyterhub &
cd shared && http-server -p 8008

