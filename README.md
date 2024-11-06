# SickKids HPC - GPU-Accelerated PyTorch Demo


### Request GPU
```
# Request 1 GPU for 5 minutes​
salloc --nodes=1 --gres=gpu:1 --time=0:05:00 -p gen_gpu​
```

### Prepare Python Interpreter
```
# Load PyTorch GPU packages​
module load pytorch/2.0.0-conda3.7-GPU

# Launch Python interpreter
python
```
### Timing Experiment 1. Matrix Multiplication
```
# Load libraries
import time​
import torch​

# Check if GPU is recognized
assert torch.cuda.is_available()

# Define matrix size
matrix_size = 1000

# Create random matrices
A = torch.randn(matrix_size, matrix_size)
B = torch.randn(matrix_size, matrix_size)

# Function to perform matrix multiplication and time it
def time_matrix_mult(device):
    A_device = A.to(device)
    B_device = B.to(device)
    start_time = time.time()
    C_device = torch.matmul(A_device, B_device)
    torch.cuda.synchronize() if device == 'cuda' else None  # Ensure all operations are finished
    end_time = time.time()
    return end_time - start_time

# Time on CPU
print(f"CPU Time: {time_matrix_mult('cpu'):.4f} seconds")

# Run once to prime GPU
time_matrix_mult('cuda')

# Time on GPU
print(f"GPU Time: {time_matrix_mult('cuda'):.4f} seconds")
```

### Timing Experiment 2. Neural Network Inference
```
# Load libraries
import torch
from torchvision.models import efficientnet_b0
import time

# Load the pre-trained EfficientNetB0 model
model = efficientnet_b0()
model.eval()  # Set the model to evaluation mode

# Create a random image tensor with shape (1, 3, 224, 224)
img = torch.randn(1, 3, 224, 224)

# Function to perform inference and time it
@torch.no_grad()
def time_forward(device, model, img):
    model = model.to(device)
    img = img.to(device)
    start_time = time.time()
    model(img)
    torch.cuda.synchronize() if device == 'cuda' else None  # Ensure all operations are finished
    end_time = time.time()
    return end_time - start_time

# Time on CPU
print(f"CPU Time: {time_forward('cpu', model, img):.4f} seconds")

# Run once to prime GPU
time_forward('cuda', model, img)

# Time on GPU
print(f"GPU Time: {time_forward('cuda', model, img):.4f} seconds")
```
