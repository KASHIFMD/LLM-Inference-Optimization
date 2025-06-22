# LLM Inference Optimization (Triton-Inference Server)
### Setup steps:
1. Triton server doc:
    ```bash
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Popular_Models_Guide/Llava1.5/llava_trtllm_guide.html#launch-triton-tensorrt-llm-container
    ```

2. Clone TensorRT-LLM Backend repository (Release tag: v0.18.2):
    ```bash
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch v0.18.2
    # Update the submodules
    cd tensorrtllm_backend
    # Install git-lfs if needed
    apt-get update && apt-get install git-lfs -y --no-install-recommends
    git lfs install
    git submodule update --init --recursive
    ```

    LINK: `https://github.com/triton-inference-server/tensorrtllm_backend/tags`

3. Create folders to be sync with Triton server:
    ```bash
    mkdir models
    mkdir tutorials
    mkdir docker_scripts
    ```

4. Launch Triton docker container with TensorRT-LLM backend:
    ```bash
    sudo docker run -it --net host --shm-size=2g \
        --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=MIG-60d6a708-f1b0-54b0-83a0-2b396432327b,MIG-71db94d9-df33-5a82-82b8-76c07dfbc45b"' \
        -v /home/ubuntu/kashif/llava_triton/tensorrtllm_backend:/tensorrtllm_backend \
        -v /home/ubuntu/kashif/llava_triton/models:/models \
        -v /home/ubuntu/kashif/llava_triton/tutorials:/tutorials \
        -v /home/ubuntu/kashif/llava_triton/sync/docker_scripts:/docker_scripts \
        -e MODEL_NAME_Llava_7b=llava-1.5-7b-hf \
        -e MODEL_NAME_Llama_11b=Llama-3.2-11B-Vision-Instruct \
        nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3
    ```

    LINK: `https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags`

5. Download Huggingface model weights:
    ```bash
    export MODEL_NAME_Llama_11b="Llama-3.2-11B-Vision-Instruct" # also Llama-3.2-11B-Vision-Instruct
    git clone https://huggingface.co/meta-llama/${MODEL_NAME_Llama_11b} models/hf_models/${MODEL_NAME_Llama_11b}

    cd /
    ```

6. Convert HuggingFace Checkpoints to TRT-LLM Format (for LLaMA Base of LLaVA):
    ```bash
    python /tensorrtllm_backend/tensorrt_llm/examples/mllama/convert_checkpoint.py \
        --model_dir models/hf_models/${MODEL_NAME_Llama_11b} \
        --output_dir models/trt_models/${MODEL_NAME_Llama_11b}/fp16/1-gpu \
        --dtype float16
    ```

7. Build TensorRT Engine:

    a. For Langauge Encoder (LLM Part)
    ```bash
    python -m tensorrt_llm.commands.build \
    --checkpoint_dir models/trt_models/${MODEL_NAME_Llama_11b}/fp16/1-gpu \
    --output_dir models/trt_engines/${MODEL_NAME_Llama_11b}/llm_engine/fp16/1-gpu \
    --max_num_tokens 4096 \
    --max_seq_len 2048 \
    --workers 1 \
    --gemm_plugin auto \
    --max_batch_size 4 \
    --max_encoder_input_len 6404 \
    --input_timing_cache model.cache
    ```

    b. For Visual Encoder (Vision Model Part of LLaVA)
    ```bash
    python /tensorrtllm_backend/tensorrt_llm/examples/multimodal/build_visual_engine.py --model_path models/hf_models/${MODEL_NAME_Llama_11b} --model_type mllama --output_dir models/trt_engines/${MODEL_NAME_Llama_11b}/vision_engine/fp16/1-gpu
    ```

8. Test Engines:
    Arguments:
    ```bash
    python /tensorrtllm_backend/tensorrt_llm/examples/multimodal/run.py --help
    ```

    For single image test:
    ```bash
    time python /tensorrtllm_backend/tensorrt_llm/examples/multimodal/run.py \
    --visual_engine_dir models/trt_engines/${MODEL_NAME_Llama_11b}/vision_engine/fp16/1-gpu \
    --visual_engine_name visual_encoder.engine \
    --llm_engine_dir models/trt_engines/${MODEL_NAME_Llama_11b}/llm_engine/fp16/1-gpu \
    --hf_model_dir models/hf_models/${MODEL_NAME_Llama_11b} \
    --image_path https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg \
    --input_text "You are a visual reasoning assistant. Carefully analyze the given image and return ONLY a valid JSON object with exactly three fields: \"alttag\", \"tags\", and \"description\".\n\nRequirements:\n- Return a valid JSON object.\n- \"alttag\": A single short sentence suitable as image alt text for SEO.\n- \"tags\": A comma-separated string containing ONLY 2 to 8 short, relevant keywords. Avoid repetition or abstract concepts.\n- \"description\": The longest and most detailed description of the image, covering layout, objects, colors, style, and context — as if explaining to a visually impaired person.\n\nImportant:\n- Do not add any comments or explanations.\n- Return ONLY the JSON.\n- All strings must be wrapped in double quotes.\n- Ensure the JSON is well-formed and properly closed." \
    --max_new_tokens 50 \
    --batch_size 2
    ```


    For batch level test:
    ```bash
    python /tensorrtllm_backend/tensorrt_llm/examples/multimodal/run.py \
        --max_new_tokens 500 \
        --hf_model_dir models/hf_models/${MODEL_NAME_Llama_11b} \
        --visual_engine_dir models/trt_engines/${MODEL_NAME_Llama_11b}/vision_encoder \
        --llm_engine_dir models/trt_engines/${MODEL_NAME_Llama_11b}/fp16/1-gpu \
        --image_path=https://images.jdmagicbox.com/v2/comp/bangalore/k6/080pxx80.xx80.130731154503.x4k6/catalogue/enrich-salon-jayanagar-3rd-block-bangalore-beauty-spas-for-women-po0eyovzeh.jpg \
        --input_text "\n Analyze the image and return a JSON object with the following structure:\n{\n  \"category_match\": \"Yes or No — Does this image belong to the category 'Beauty Salons'?\",\n  \"contains_human\": \"Yes or No — Is a person visible in the image?\",\n  \"visible_items\": \"Comma-separated list of key items seen in the image (e.g., chair, mirror, hair dryer, cosmetics).\",\n  \"text_detected\": \"List any readable text from signs or posters in the image.\",\n  \"description\": \"Write a detailed and descriptive paragraph about what is happening in the image, as if explaining it to someone who cannot see it, focusing on elements related to a Beauty Salon.\"\n}\nOnly return valid JSON. Do not include any extra explanation or markdown." \
        --batch_size=1 # for LLaVA
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




