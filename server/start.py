import os
# get server ip
import socket
from classes.config_loader import WorkerManager
os.environ["LOAD_MODEL"] = "True"  # Default to True, can be overridden based on environment

if __name__ == "__main__":
    os.environ["SERVER_IP"] = "A.B.C.D"
    os.environ["PROCESS"] = "image_relevancy"
    os.environ["LOAD_MODEL"] = "False"
    loadModel= True if os.environ["LOAD_MODEL"] == "True" else False
    print("Loading model:", loadModel)
    manager = WorkerManager(loadModel)
    manager.check_and_update_workers()


