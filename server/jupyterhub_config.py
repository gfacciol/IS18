#c.JupyterHub.authenticator_class = 'dummyauthenticator.DummyAuthenticator'

#import os


#c.JupyterHub.authenticator_class = 'tmpauthenticator.TmpAuthenticator'
#c.LocalAuthenticator.create_system_users = True
#c.Authenticator.auto_login = True



c.JupyterHub.authenticator_class = 'tmpauthenticator.TmpAuthenticator'



c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.Spawner.mem_limit = '2G'
c.Spawner.cpu_limit = 8.0
c.DockerSpawner.image = "facciolo/is18-satellite-minitutorial"
c.DockerSpawner.host_ip = "0.0.0.0"

# this variable indicates the external URL of the web server used for serving the point clouds
c.DockerSpawner.environment = {
    'EXTERNAL_HTTP_SRV_URL': 'http://avocat.ovh.hw.ipol.im:8008',
}
c.DockerSpawner.volumes = {
    '/home/facciolo/IS18/shared': '/shared'
}


c.Spawner.args = ['--NotebookApp.allow_origin=*']



#c.JupyterHub.hub_connect_ip = os.environ['JUPYTERHUB_SERVICE_HOST_IP']

#network_name = os.environ['DOCKER_NETWORK_NAME']
c.DockerSpawner.use_internal_ip = True
#c.DockerSpawner.network_name = network_name
# Pass the network name as argument to spawned containers
#c.DockerSpawner.extra_host_config = {
#        'network_mode': network_name,
#        'volume_driver': 'local'
#    }
c.DockerSpawner.remove_containers = True
c.DockerSpawner.debug = True
c.JupyterHub.hub_ip = "172.17.0.1"






#################################


#c.JupyterHub.spawner_class = 'systemdspawner.SystemdSpawner'
#c.SystemdSpawner.mem_limit = '2G'
##c.SystemdSpawner.cpu_limit = 8.0
#c.SystemdSpawner.disable_user_sudo = True
##c.SystemdSpawner.use_sudo = True
##
##
c.JupyterHub.hub_port = 54321

