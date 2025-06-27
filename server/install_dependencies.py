import subprocess

packages = [
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
    "tensorrt-llm",
    "python-multipart",
    "requests",
    "tqdm"
]

subprocess.run(["pip", "install"] + packages)
print("All dependencies installed successfully.")
