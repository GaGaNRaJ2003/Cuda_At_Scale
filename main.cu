#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];

        // Debug: Print intermediate values for the first few indices
        if (idx < 2) {
            printf("Thread %d: A[%d] = %f, B[%d] = %f, C[%d] = %f\n",
                   idx, idx, A[idx], idx, B[idx], idx, C[idx]);
        }
    }
}

int main() {
    int N = 1 << 20; // 1 million elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Debug: Print first few elements of h_A and h_B
    std::cout << "h_A[0] = " << h_A[0] << ", h_A[1] = " << h_A[1] << std::endl;
    std::cout << "h_B[0] = " << h_B[0] << ", h_B[1] = " << h_B[1] << std::endl;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Debug: Copy data back to host and print
    float* temp_A = new float[N];
    float* temp_B = new float[N];
    cudaMemcpy(temp_A, d_A, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_B, d_B, bytes, cudaMemcpyDeviceToHost);

    std::cout << "d_A[0] = " << temp_A[0] << ", d_A[1] = " << temp_A[1] << std::endl;
    std::cout << "d_B[0] = " << temp_B[0] << ", d_B[1] = " << temp_B[1] << std::endl;

    delete[] temp_A;
    delete[] temp_B;

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "Blocks per grid: " << blocksPerGrid << std::endl;

    // Launch the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cout << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}