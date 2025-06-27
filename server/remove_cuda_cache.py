import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-60d6a708-f1b0-54b0-83a0-2b396432327b,MIG-71db94d9-df33-5a82-82b8-76c07dfbc45b"
# device = torch.device("cuda:<device_id>")
# with torch.cuda.device(device):
torch.cuda.empty_cache()
