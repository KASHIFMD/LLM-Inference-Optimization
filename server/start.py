import os
# get server ip
import socket
from classes.config_loader import WorkerManager
os.environ["LOAD_MODEL"] = "True"  # Default to True, can be overridden based on environment

if __name__ == "__main__":
    os.environ["SERVER_IP"] = "103.42.50.106"
    os.environ["PROCESS"] = "image_relevancy"
    
    
    # hostname = socket.gethostname()
    # IPAddr = socket.gethostbyname(hostname)
    # print(f"Server IP Address: {IPAddr}")
    # # Check if the server is running in a Docker container
    # ignoreModelLoadIps = ['172.17.0.14', '192.168.24.151', '192.168.40.172']
    # ModelLoadIps =  ['10.1.1.94', '10.1.1.52', '172.17.0.3']
    # if IPAddr in ignoreModelLoadIps or IPAddr not in ModelLoadIps:
    #     #check if the docker running in cpu server
    #     print("Running in CPU server, not loading model")
    os.environ["LOAD_MODEL"] = "False"
    loadModel= True if os.environ["LOAD_MODEL"] == "True" else False
    print("Loading model:", loadModel)
    manager = WorkerManager(loadModel)
    manager.check_and_update_workers()


