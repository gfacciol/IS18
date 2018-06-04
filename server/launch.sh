cd /home/facciolo/IS18
#sudo jupyterhub --ssl-key ./server/conf/live/avocat.ovh.hw.ipol.im/privkey.pem --ssl-cert ./server/conf/live/avocat.ovh.hw.ipol.im/fullchain.pem &
jupyterhub &
cd shared && http-server -p 8008 &

