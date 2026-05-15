import torch
import time


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch.cuda.synchronize
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps", torch.mps.synchronize
    return torch.device("cpu"), "cpu", lambda: None


def check_acceleration():
    device, name, sync = pick_device()
    if name == "cpu":
        print("⚠️  No GPU backend found (neither CUDA nor MPS). Running on CPU.")
    else:
        label = "NVIDIA GPU" if name == "cuda" else "Apple GPU"
        print(f"✅ Success! Running on {label} via {device}")

    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)

    start = time.time()
    torch.matmul(x, y)
    sync()
    end = time.time()

    print(f"⚡ 5000x5000 Matrix Multiplication took: {end - start:.4f} seconds")


if __name__ == "__main__":
    check_acceleration()