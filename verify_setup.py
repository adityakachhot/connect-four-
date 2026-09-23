"""
Setup verification script for RL project on Apple Silicon (M3 Air).
Confirms Python version, package installations, and PyTorch MPS (Metal Performance Shaders) backend availability.
"""
import sys
import platform

def verify_environment():
    print(f"Python Version: {platform.python_version()} ({sys.executable})")
    assert sys.version_info >= (3, 10), "Python 3.10+ is required."

    # Check numpy
    import numpy as np
    print(f"NumPy Version: {np.__version__}")

    # Check tqdm
    import tqdm
    print(f"tqdm Version: {tqdm.__version__}")

    # Check matplotlib
    import matplotlib
    print(f"Matplotlib Version: {matplotlib.__version__}")

    # Check torch & MPS
    import torch
    print(f"PyTorch Version: {torch.__version__}")

    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()

    print(f"torch.backends.mps.is_built(): {mps_built}")
    print(f"torch.backends.mps.is_available(): {mps_available}")

    if mps_available:
        device = torch.device("mps")
        x = torch.ones(5, device=device)
        y = x * 2
        print(f"MPS Tensor operation test: {y.cpu().numpy()} (device={y.device})")
        print("MPS backend is operational!")
    else:
        print("WARNING: MPS backend is NOT available.")

    print("\nEnvironment verification successful!")

if __name__ == "__main__":
    verify_environment()
