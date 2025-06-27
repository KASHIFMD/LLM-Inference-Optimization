import yaml
import os
import time
import subprocess
import hashlib
import signal
import copy

CONFIG_PATH = "/server/worker_config.yml"  # Update with your file path
child_process = None
class WorkerManager:
    def __init__(self, loadModel, config_path=CONFIG_PATH):
        self.loadModel = loadModel
        self.config_path = config_path
        self.last_config_hash = None
        self._prev_env_vars = {}

    def load_worker_config(self):
        """
        Load worker configuration from the YAML file.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return None

    def validate_worker_config(self):
        """
        Validate the structure and content of the worker configuration file.
        """
        config = self.load_worker_config()
        if not config:
            print("Error: Configuration file is empty or corrupted.")
            return False

        required_keys = ["servers"]
        gpu_server_keys = ["name", "host", "port", "containers"]
        container_keys = ["name", "image", "workers", "threads", "gpu_limit", "environment", "restart_policy"]

        # Check top-level keys
        for key in required_keys:
            if key not in config:
                print(f"Error: Missing key '{key}' in the configuration.")
                return False

        # Check GPU servers
        gpu_servers = config.get("servers", {}).get("gpu_servers", [])
        if not isinstance(gpu_servers, list):
            print("Error: 'gpu_servers' should be a list.")
            return False

        for server in gpu_servers:
            for key in gpu_server_keys:
                if key not in server:
                    print(f"Error: Missing key '{key}' in GPU server configuration: {server}")
                    return False
                if key == "port" and not isinstance(server[key], int):
                    print(f"Error: 'port' should be an integer. Found: {server[key]}")
                    return False

            containers = server.get("containers", [])
            if not isinstance(containers, list):
                print(f"Error: 'containers' should be a list in server {server['name']}.")
                return False

            for container in containers:
                for key in container_keys:
                    if key not in container:
                        print(f"Error: Missing key '{key}' in container configuration: {container}")
                        return False
                    if key == "workers" and not isinstance(container[key], int):
                        print(f"Error: 'workers' should be an integer in container {container['name']}.")
                        return False
                    if key == "threads" and not isinstance(container[key], int):
                        print(f"Error: 'threads' should be an integer in container {container['name']}.")
                        return False
                    if key == "gpu_limit" and not isinstance(container[key], int):
                        print(f"Error: 'gpu_limit' should be an integer in container {container['name']}.")
                        return False
                    if key == "environment" and not isinstance(container[key], dict):
                        print(f"Error: 'environment' should be a dictionary in container {container['name']}.")
                        return False

        print("Configuration validation successful.")
        return True

    def get_worker_count(self, container_name):
        """
        Get the worker count for a specific container.
        """
        config = self.load_worker_config()
        if config:
            for server in config.get("servers", {}).get("gpu_servers", []):
                for container in server.get("containers", []):
                    if container["name"] == container_name:
                        return container.get("workers", 1)  # Default to 1 if not found
        return 1

    def get_container_config(self, container_name):
        """
        Get the full configuration for a specific container.
        """
        config = self.load_worker_config()
        if config:
            for server in config.get("servers", {}).get("gpu_servers", []):
                for container in server.get("containers", []):
                    if container["name"] == container_name:
                        return container
        return None

    def calculate_config_hash(self):
        """
        Calculate the hash of the config file for change detection.
        """
        try:
            with open(self.config_path, 'rb') as file:
                data = file.read()
                return hashlib.md5(data).hexdigest()
        except Exception as e:
            print(f"Error calculating hash: {e}")
            return None

    def restart_container(self, container_name, workers, max_tries=3):
        global child_process
        # Step 1: Kill existing child process if it's running
        if child_process is not None and child_process.poll() is None:
            try:
                os.kill(child_process.pid, signal.SIGTERM)
                print(f"Child process {child_process.pid} terminated.")
            except ProcessLookupError:
                print(f"Child process {child_process.pid} not found. Might have already exited.")
            except Exception as e:
                print(f"Error terminating child process: {e}")
            finally:
                child_process = None

        print(f"Restarting container {container_name} with {workers} workers.")
        # Retry logic for starting Uvicorn
        for attempt in range(1, max_tries + 1):
            print(f"### Attempt {attempt} to start Uvicorn server with {workers} workers...")
            try:
                # Start the child process running "python3 main.py"
                print(f"Starting Uvicorn server with {workers} workers...")
                child_process = subprocess.Popen(["python3", "app.py", "--workers", str(workers)])
                time.sleep(2**attempt)  # Exponential backoff (2, 4, 8 seconds)

                # Check if Uvicorn started successfully
                if child_process.poll() is None:
                    print(f"Successfully started {container_name} with {workers} workers on attempt {attempt}. \n\n\n")
                    return
                else:
                    print(f"Failed to start Uvicorn server on attempt {attempt}. Retrying...\n\n\n")
            except Exception as e:
                print(f"Error starting Uvicorn on attempt {attempt}: {e}, \n\n\n")

        # If all attempts fail
        print(f"Failed to start Uvicorn server after {max_tries} attempts. \n\n\n")

    def update_environment_variables(self, prev_env_dict, container_key, current_env):
        prev_env = prev_env_dict.get(container_key, {})
        isUpdateNeeded = False
        # Check if any environment variable has changed
        for key, value in current_env.items():
            if prev_env.get(key) != value:
                env_key = container_key + "_" + key
                if isinstance(value, dict):
                    updated = self.update_environment_variables(prev_env, env_key, value)
                    isUpdateNeeded = isUpdateNeeded or updated
                    continue                
                print("### ", env_key, ":", value)
                os.environ[env_key] = str(value)
                print(f"[Env Update] {env_key} updated to: {value}")
                isUpdateNeeded = True

        # Update the cache
        prev_env_dict[container_key] = copy.deepcopy(current_env)
        return isUpdateNeeded

    def check_and_update_workers(self):
        """
        Continuously check for configuration changes and update workers.
        """
        print("Starting worker monitoring...")
        while True:
            current_hash = self.calculate_config_hash()
            
            if current_hash and current_hash != self.last_config_hash:
                print("Configuration change detected, updating workers...")
                self.last_config_hash = current_hash

                if not self.validate_worker_config():
                    print("Invalid configuration. Skipping update.")
                    time.sleep(5)
                    continue
                
                config = self.load_worker_config()
                if config:
                    for server_type in ["gpu_servers", "cpu_servers"]:
                        for server in config.get("servers", {}).get(server_type, []):
                            for current_container in server.get("containers", []):                    
                                container_key = f"{server['host']}_{current_container['process']}"
                                self._prev_env_vars.setdefault(container_key, {})
                                isUpdateNeeded = self.update_environment_variables(self._prev_env_vars, container_key, current_container)
                                if(isUpdateNeeded):
                                    container_name = current_container["name"]
                                    workers = current_container.get("workers", 1)
                                    
                                    print("SERVER_IP:", os.environ.get("SERVER_IP"))
                                    print("server host:", server.get("host", ""))
                                    print("PROCESS:", os.environ.get("PROCESS"))
                                    print("container process:", current_container.get("process", ""))

                                    if(os.environ["SERVER_IP"] == server.get("host", "") and os.environ["PROCESS"] == current_container.get("process", "")):
                                        print("workers: ", workers)
                                        print("Restarting current_container: ", container_name)
                                        self.restart_container(container_name, workers)
                print("Worker update completed.")
            else:
                print(".", end="")  # No change detected, do nothing
            time.sleep(5)  # Check every 5 seconds
            
    def check_and_update_workers_2(self):
        """
        Continuously check for configuration changes and update workers.
        """
        print("Starting worker monitoring...")
        while True:
            current_hash = self.calculate_config_hash()
            
            if current_hash and current_hash != self.last_config_hash:
                print("Configuration change detected, updating workers...")
                self.last_config_hash = current_hash

                if not self.validate_worker_config():
                    print("Invalid configuration. Skipping update.")
                    time.sleep(5)
                    continue
                os.environ["GPU_SERVER_IP_RELEVANCY"] = ""
                config = self.load_worker_config()
                if config and self.loadModel:
                    for server in config.get("servers", {}).get("gpu_servers", []):
                        for container in server.get("containers", []):
                            if(server.get("host", "") == os.environ["GPU_SERVER_IP"] and container.get("process", "") == os.environ["PROCESS"]):
                                gpu_limit = container.get("gpu_limit", 4)
                                os.environ["GPU_LIMIT_PER_PROCESS"] = str(gpu_limit)

                                env = container.get("environment", {})
                                if isinstance(env, dict):
                                    for key, value in env.items():
                                        os.environ[key] = value
                                container_name = container["process"]
                                workers = container.get("workers", 1)
                                self.restart_container(container_name, workers)
                            if(server.get("name", "") == "GPU_Server_2" and container.get("process", "") == "image_relevancy"):
                                
                                gpu_limit = container.get("gpu_limit", 4)
                                os.environ["GPU_LIMIT_PER_PROCESS"] = str(gpu_limit)

                                env = container.get("environment", {})
                                if isinstance(env, dict):
                                    for key, value in env.items():
                                        os.environ[key] = value
                                container_name = container["process"]
                                workers = container.get("workers", 1)
                                print("Restarting container: ", container_name)
                                
                                self.restart_container(container_name, workers)
                    os.environ["GPU_SERVER_IP_RELEVANCY"] = os.environ.get("GPU_SERVER_IP_RELEVANCY", " ")[:-1]
                elif config and not self.loadModel:
                    for server in config.get("servers", {}).get("cpu_servers", []):
                        for container in server.get("containers", []):
                            if(server.get("host", "") == os.environ["CPU_SERVER_IP"] and container.get("process", "") == "content_image_optimization"):
                                container_name = container["process"]
                                workers = container.get("workers", 1)
                                
                                print("workers: ", workers)
                                print("type(workers): ", type(workers))
                                if workers>0:
                                    os.environ["GPU_SERVER_IP_RELEVANCY"] =  server.get("host", "")+ ","
                                    os.environ["GPU_SERVER_PORT_RELEVANCY"] = container.get("port", "")
                                print("GPU_SERVER_IP_RELEVANCY: ", os.environ["GPU_SERVER_IP_RELEVANCY"])
                                print("GPU_SERVER_PORT_RELEVANCY: ", os.environ["GPU_SERVER_PORT_RELEVANCY"])
                                self.restart_container(container_name, workers)
                print("Worker update completed.")
            else:
                print(".", end=" ")  # No change detected, do nothing
            time.sleep(5)  # Check every 5 seconds
