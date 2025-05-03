
---

# CUDA Vector Addition at Scale

![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen) ![C++](https://img.shields.io/badge/Language-C++-blue)

This is a simple yet efficient implementation of **vector addition** using CUDA, designed to handle large-scale data arrays with millions of elements. The program leverages GPU parallelism to perform element-wise addition of two vectors and verifies the correctness of the results.

---

## Features

- **GPU Acceleration**: Utilizes CUDA to perform parallel computations on NVIDIA GPUs.
- **Scalable**: Handles vectors with up to **1 million elements** (or more, depending on GPU memory).
- **Built-in Verification**: Ensures correctness by comparing computed results with expected values.
- **Easy to Run**: Works locally or in cloud environments like Google Colab.

---

## Requirements

### Hardware
- A CUDA-capable GPU (e.g., NVIDIA Tesla, GeForce, or RTX series).

### Software
- **CUDA Toolkit**: Install the version compatible with your GPU's compute capability ([check here](https://developer.nvidia.com/cuda-gpus)).
- **C++ Compiler**: GCC or Clang with CUDA support.

Optional:
- **Google Colab**: No local setup required—run directly in the cloud.

---

## Installation and Usage

### Local Setup
1. **Install CUDA Toolkit**:
   Follow the [official CUDA installation guide](https://developer.nvidia.com/cuda-downloads).

2. **Compile the Code**:
   ```bash
   nvcc main.cu -o vector_add -arch=sm_75  # Replace sm_75 with your GPU's compute capability
   ```

3. **Run the Program**:
   ```bash
   ./vector_add
   ```

### Google Colab Setup
1. Open a new [Google Colab notebook](https://colab.research.google.com/).
2. Upload `main.cu` to the Colab environment.
3. Compile and run:
   ```bash
   !nvcc main.cu -o vector_add -arch=sm_75  # Adjust compute capability as needed
   !./vector_add
   ```

---

## How It Works

1. Two vectors (`A` and `B`) are initialized on the host:
   - `A[i] = i`
   - `B[i] = 2 * i`

2. The vectors are transferred to the GPU memory.

3. A CUDA kernel performs element-wise addition:
   ```cpp
   C[i] = A[i] + B[i]
   ```

4. The result vector `C` is copied back to the host and verified for correctness.

Example:
```
A = [0, 1, 2, 3]
B = [0, 2, 4, 6]
C = [0, 3, 6, 9]
```

If all elements match, the program outputs:
```
Vector addition successful!
```

---

## Performance

- On an NVIDIA Tesla T4 GPU:
  - **1 million elements**: Completed in ~2 ms.
- Performance scales linearly with the number of GPU cores.

---

## License

This project is released under the **MIT License**. Feel free to use and modify it as needed.

---

### Acknowledgments

Thanks to NVIDIA for providing the CUDA platform and enabling GPU-accelerated computing.

---
