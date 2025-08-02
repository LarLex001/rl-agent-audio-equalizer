import numpy as np
import torch

np.set_printoptions(threshold=np.inf)

print("PyTorch version:", torch.__version__)

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
else:
    print("No CUDA devices found, using CPU")

print("CPU is available:", torch.backends.mps.is_available() if torch.backends.mps.is_available() else "Using CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)