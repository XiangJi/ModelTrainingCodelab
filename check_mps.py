import torch
import time

def check_m4_acceleration():
    if not torch.backends.mps.is_available():
        print("❌ MPS (Metal Performance Shaders) not found. Check your PyTorch install.")
        return

    device = torch.device("mps")
    print(f"✅ Success! Running on M4 GPU via {device}")

    # Simple benchmark
    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)
    
    start = time.time()
    torch.matmul(x, y) # Force GPU computation
    torch.mps.synchronize() # Ensure MPS operations complete
    end = time.time()
    
    print(f"⚡ 5000x5000 Matrix Multiplication took: {end - start:.4f} seconds")

if __name__ == "__main__":
    check_m4_acceleration()