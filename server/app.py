import sys
sys.path.append("/tensorrtllm_backend/tensorrt_llm")
from fastapi.responses import JSONResponse
import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import subprocess
import tensorrt_llm
from tensorrt_llm.runtime import MultimodalModelRunner
from examples.utils import add_common_args
import uvicorn
import json

app = FastAPI()

model_runner = None
args = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_runner, args
    SERVER_IP = os.environ.get("SERVER_IP", "localhost")
    PROCESS = os.environ.get("PROCESS", "content_image_optimization")
    workers = int(os.environ.get(f"{SERVER_IP}_{PROCESS}_workers"))
    if(workers!=0):
        print("🔄 🔄 🔄 🔄 🔄 🔄 🔄 🔄 🔄 🔄 🔄 🔄 Loading Llama3.2-vision model...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        args = get_args()
        model_runner = MultimodalModelRunner(args)
        print("✅ Model loaded successfully.")
    else:
        print("The model won't be loaded as the worker count is 0.")
    yield
    print("🛑 Shutting down model...")

app = FastAPI(lifespan=lifespan)

class InferenceRequest(BaseModel):
    image_url: str
    is_json: Optional[int] = 0  # Default is False, indicating no JSON response
    prompt: Optional[str] = ("You are a visual reasoning assistant. Carefully analyze the given image and "
        "return ONLY a valid JSON object with exactly three fields: \"alttag\", \"tags\", and \"description\".\n\n"
        "Requirements:\n- Return a valid JSON object.\n- \"alttag\": A single short sentence suitable as image alt text for SEO.\n"
        "- \"tags\": A comma-separated string containing ONLY 2 to 8 short, relevant keywords. Avoid repetition or abstract concepts.\n"
        "- \"description\": The longest and most detailed description of the image, covering layout, objects, colors, style, and context — "
        "as if explaining to a visually impaired person.\n\nImportant:\n- Do not add any comments or explanations.\n"
        "- Return ONLY the JSON.\n- All strings must be wrapped in double quotes.\n- Ensure the JSON is well-formed and properly closed.")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Llama3.2 Vision API is running."}
@app.post("/v1/infer")
def infer(req: InferenceRequest):
    print("req: ", req)
    global model_runner, args
    retries = 3 if req.is_json==1 else 1
    output_text = [[""]]  # Predefine to avoid UnboundLocalError

    for attempt in range(1, retries + 1):
        try:
            input_data = model_runner.load_test_data(req.image_url, None)
            print("req.prompt: ", req.prompt)
            input_text, output_text = model_runner.run(req.prompt, input_data, args.max_new_tokens)

            # Apply cleaned_res logic
            raw_output = output_text[0][0]
            print(f"Raw output from model:\n{raw_output}\n")
            cleaned_res = raw_output  # No replacement logic applied
            print(f"[Attempt={attempt}] cleaned_res: {cleaned_res}")

            # Validate as JSON
            if req.is_json == 1:
                json.loads(cleaned_res)  # Will raise if invalid

            print(f"[Final: Attempt={attempt}] cleaned_res: {cleaned_res}")
            return JSONResponse(
                status_code=200,
                content={
                    "error_code": 0,
                    "err": "",
                    "message": "Image Processed Successfully",
                    "input": req.prompt,
                    "image_url": req.image_url,
                    # "cat_name": "",
                    "output": cleaned_res
                }
            )

        except json.JSONDecodeError:
            if attempt < retries:
                print(f"[Attempt {attempt}] Invalid JSON, retrying...\nOutput:\n{cleaned_res}\n")
            else:
                print(f"[Attempt {attempt}] Still invalid. Returning last response.")
                return JSONResponse(
                    status_code=200,
                    content={
                        "error_code": 2,
                        "err": "Invalid JSON after retries",
                        "message": "Image Processed Successfully",
                        "input": req.prompt,
                        "image_url": req.image_url,
                        "cat_name": "",
                        "output": cleaned_res
                    }
                )

        except Exception as e:
            print(f"[Attempt {attempt}] Exception occurred: {e}")
            if attempt >= retries:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error_code": 1,
                        "err": str(e),
                        "message": "Processing Failed",
                        "input": req.prompt,
                        "image_url": req.image_url,
                        "cat_name": "",
                        "output": ""
                    }
                )
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    MODEL_NAME_Llama_11b = os.getenv("MODEL_NAME_Llama_11b")
    parser.set_defaults(
        visual_engine_dir=f"/models/trt_engines/multimodal/{MODEL_NAME_Llama_11b}/encoder",
        visual_engine_name="visual_encoder.engine",
        llm_engine_dir=f"/models/trt_engines/multimodal/{MODEL_NAME_Llama_11b}/decoder",
        hf_model_dir=f"/models/hf_models/{MODEL_NAME_Llama_11b}/",
        max_new_tokens=800,
        batch_size=2,
        log_level="INFO"
    )
    return parser.parse_args([])

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    SERVER_IP = os.environ.get("SERVER_IP", "localhost")
    PROCESS = os.environ.get("PROCESS", "content_image_optimization")
    print("SERVER_IP: ", SERVER_IP)
    print("PROCESS: ", PROCESS)
    
    # Load the worker configuration
    port = int(os.environ[f"{SERVER_IP}_{PROCESS}_port"])
    workers = int(os.environ.get(f"{SERVER_IP}_{PROCESS}_workers", 1))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=workers)
