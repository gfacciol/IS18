cd /home/student1 && jupyter notebook --port=8000 --ip=* --allow-root --NotebookApp.token='' & 
cd /shared && http-server -p 8008 
