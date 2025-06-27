# LLM Inference Optimization (Triton-Inference Server)

Refer my technical [Blogs](https://kashifmd.github.io/blogs/) for some standard methods of LLM inference optimization

### Setup steps for Llama3.2 vision model with triton inference server:
1. The reference Triton server documentation for the model Llava1.5v-7b model:
    ```bash
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Popular_Models_Guide/Llava1.5/llava_trtllm_guide.html#launch-triton-tensorrt-llm-container
    ```
    But don't worry, go through following steps for hosting Llama3.2 vision model for inferencing.
2. Clone TensorRT-LLM Backend repository (Release tag: v0.18.2):
    ```bash
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch v0.18.2
    # Update the submodules
    cd tensorrtllm_backend
    # Install git-lfs if needed, it install git-lfs client (Git Large File Storage) is an open-source extension for Git that handles large files by replacing them with text pointers inside Git, while the actual file contents are stored on a remote server.
    apt-get update && apt-get install git-lfs -y --no-install-recommends
    # it initializes Git LFS for your current user account.
    git lfs install
    # it cloned/initializes, updates all the submodules recursively
    git submodule update --init --recursive
    ```

    Change `v0.18.2` with the required version that you want, be sure to have latest version if setting for the first time.
    
    To check latest version of TensorRT-LLM backend git repo
    LINK: `https://github.com/triton-inference-server/tensorrtllm_backend/tags`

3. Create folders to be sync with Triton server:

    clone the tutorials [tutorials](https://github.com/triton-inference-server/tutorials) repo of triton 
    ```bash
    git clone https://github.com/triton-inference-server/tutorials
    ```
    ```bash
    # create `docker_scripts` to sync as volumes in triton container, for some other files to sync. `server` will be used for FastAPI/flask-app apis:
    mkdir docker_scripts
    mkdir server
    ```

4. Download Huggingface model weights:
    ```bash
    export MODEL_NAME_Llama_11b="Llama-3.2-11B-Vision-Instruct" # also Llama-3.2-11B-Vision-Instruct
    git clone https://huggingface.co/meta-llama/${MODEL_NAME_Llama_11b} models/hf_models/${MODEL_NAME_Llama_11b}
    cd /
    ```
    You will need huggingface `username` and `password` for this, on huggingface website create your account.

5. Launch Triton docker container with TensorRT-LLM backend:
    ```bash
    sudo docker run -it --net host --shm-size=2g \
        --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=MIG-ID-01,MIG-ID-02"' \
        -v /rel_path_triton_server/tensorrtllm_backend:/tensorrtllm_backend \
        -v /rel_path_triton_server/models:/models \
        -v /rel_path_triton_server/tutorials:/tutorials \
        -v /rel_path_triton_server/docker_scripts:/docker_scripts \
        -v /rel_path_triton_server/server:/server   \
        -e MODEL_NAME_Llava_7b=llava-1.5-7b-hf \
        -e MODEL_NAME_Llama_11b=Llama-3.2-11B-Vision-Instruct \
        nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3
    ```
    Change the MIG-ID-01 and MIG-ID-02 ids with your GPU ids. if there is no GPU partition remove `'"device=MIG-ID-01,MIG-ID-02"'` part.

    Change `nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3` with the tag you want for your Triton server container (make it latest if using for the first time)

    LINK: `https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags`

    - `-it`: Allows you to interact with the container via terminal — like you would with an SSH session. You can use `-d` to make terminal to be detachable.
    - `--net host`: Uses the host machine’s network stack directly inside the container.
    - `--shm-size=2g`: Sets the shared memory size (/dev/shm) to 2 GB. Avoids memory errors for apps (like PyTorch, TensorRT, OpenCV) that rely on shared memory.
    - `--ulimit memlock=-1`: Sets the maximum locked-in-memory address space to unlimited. Allows processes in the container to lock memory (prevent it from being swapped to disk). Required for performance-critical apps like GPU inference engines.
    - `--ulimit stack=67108864`: Sets the stack size limit to 64 MB (in bytes). Prevents stack overflow errors in deep recursion or heavy multithreaded GPU workloads.
    - `-v`: Mounts directories from host to container (bind mounts)
    - `-e`: Sets environment variables (e.g., MODEL_NAME_Llava_7b)

6. Set environment variables:
    ```bash
    export HF_MODEL_PATH_LLAMA=/models/hf_models/${MODEL_NAME_Llama_11b}/
    export UNIFIED_CKPT_PATH=/models/trt_models/${MODEL_NAME_Llama_11b}/fp16/1-gpu
    export ENGINE_PATH=/models/trt_engines/${MODEL_NAME_Llama_11b}/fp16/1-gpu
    export MULTIMODAL_ENGINE_PATH=/models/trt_engines/${MODEL_NAME_Llama_11b}/multimodal_encoder
    export BUILD_VISUAL=/tensorrtllm_backend/tensorrt_llm/examples/multimodal/build_visual_engine.py
    export CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/mllama/convert_checkpoint.py
    export LLAMA_ENCODER_ENGINE=/models/trt_engines/multimodal/${MODEL_NAME_Llama_11b}/encoder
    export LLAMA_DECODER_ENGINE=/models/trt_engines/multimodal/${MODEL_NAME_Llama_11b}/decoder
    export LLAMA_CHECKPOINTS=/models/trt_ckpts/multimodal/${MODEL_NAME_Llama_11b}
    export RUN_CODE=/tensorrtllm_backend/tensorrt_llm/examples/multimodal/run.py
    ```

7. Build engine of vision encoder part of the model:

    execute for the first time only,
    ```bash
    time python ${BUILD_VISUAL} \
        --model_type mllama \
        --model_path ${HF_MODEL_PATH_LLAMA} \
        --output_dir ${LLAMA_ENCODER_ENGINE}
    ```

8. Build engine of decoder part of the model:
    
    a. Prepare the model for engine building by restructuring weights, quantizing or casting to appropriate data types, and making them TensorRT-compatible. Converts the HuggingFace-format language model weights (decoder) into a format that can be used by TensorRT-LLM. 
    
    Execute for the first time only,

    ```bash
    time python ${CONVERT_CHKPT_SCRIPT} \
        --model_dir ${HF_MODEL_PATH_LLAMA} \
        --output_dir ${LLAMA_CHECKPOINTS} \
        --dtype bfloat16
    ```

    b. Takes the converted decoder weights and compiles them into a TensorRT engine for fast GPU inference. Includes optimizing matrix operations, memory layout, and GPU kernels for decoding tasks.

    - Speeds up text generation during inference dramatically.

    - Tailors the decoder model to available GPU capabilities and runtime constraints (batch size, sequence length, etc.).

    - This engine is used repeatedly for serving inference requests with minimal latency.

    ```bash
    time python3 -m tensorrt_llm.commands.build \
        --checkpoint_dir ${LLAMA_CHECKPOINTS} \
        --output_dir ${LLAMA_DECODER_ENGINE} \
        --max_num_tokens 4096 \
        --max_seq_len 2048 \
        --workers 1 \
        --gemm_plugin auto \
        --max_batch_size 4 \
        --max_encoder_input_len 6404 \
        --input_timing_cache model.cache
    ```

9. Make inference using ${RUN_CODE}:

    ```bash
    time python3 ${RUN_CODE} \
        --visual_engine_dir ${LLAMA_ENCODER_ENGINE} \
        --visual_engine_name visual_encoder.engine \
        --llm_engine_dir ${LLAMA_DECODER_ENGINE} \
        --hf_model_dir ${HF_MODEL_PATH_LLAMA} \
        --image_path https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg \
        --input_text "You are a visual reasoning assistant. Carefully analyze the given image and return ONLY a valid JSON object with exactly three fields: \"alttag\", \"tags\", and \"description\".\n\nRequirements:\n- Return a valid JSON object.\n- \"alttag\": A single short sentence suitable as image alt text for SEO.\n- \"tags\": A comma-separated string containing ONLY 2 to 8 short, relevant keywords. Avoid repetition or abstract concepts.\n- \"description\": The longest and most detailed description of the image, covering layout, objects, colors, style, and context — as if explaining to a visually impaired person.\n\nImportant:\n- Do not add any comments or explanations.\n- Return ONLY the JSON.\n- All strings must be wrapped in double quotes.\n- Ensure the JSON is well-formed and properly closed." \
        --max_new_tokens 800 \
        --batch_size 2
    ```


# LLM Inference Optimization (CPU-focused Experiments)

This repository contains practical experiments to optimize Large Language Model (LLM) inference without using GPUs. It is designed to help developers, researchers, and engineers understand trade-offs between precision, speed, and memory usage on local machines.

## 🔍 Focus Areas
- Quantization (e.g., FP32 → INT8)
- Model size vs performance
- Token generation latency
- Peak memory usage
- ONNX, GGUF, and CPU-based runtime libraries

## 📁 Folder Structure
```
llm-inference-optimization/
├── notebooks/              # Jupyter notebooks for experiments
├── scripts/                # Utility scripts for benchmarking
├── models/                 # Model configs or converted formats
├── results/                # Logs and output from experiments
└── README.md
```

## 📦 Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ✅ Requirements (included in requirements.txt)
- transformers
- optimum
- onnxruntime
- psutil
- matplotlib
- tqdm

## 🚀 Experiments Included
1. **Quantization**: Compare full precision vs int8 using `optimum`
2. **Latency Benchmark**: Measure time to generate N tokens on CPU
3. **Memory Footprint**: Track peak memory usage across model sizes
4. **Conversion to ONNX**: Use `optimum.exporters` to benchmark ONNX

## 📓 Sample Notebook
Start with `notebooks/01_quantization_vs_fp32.ipynb` for a direct comparison of inference performance.

## 🤝 Contributing
If you experiment with new techniques or tools (e.g., GGUF, llama.cpp), feel free to open a PR.

---

# 🧪 Notebook: 01_quantization_vs_fp32.ipynb
## Objective:
Compare inference latency and memory of BERT-base in FP32 vs INT8 using Hugging Face + Optimum.

### Steps:
- Load `bert-base-uncased`
- Use `optimum` to quantize to INT8
- Run dummy inference with a sample input
- Measure latency (time) and memory (psutil)
- Visualize comparison

Want the full notebook code scaffold too?

# LLM Inference Optimization (GPU-focused Experiments)




