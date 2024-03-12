#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1
//STEP 1

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float* C, const float* A, const float* B, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float val = 0.0f;
        for (int k = 0; k < size; ++k) {
            val += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = val;
    }
}

// Host function
void matrixMulHost(float* C, const float* A, const float* B, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float val = 0.0f;
            for (int k = 0; k < size; ++k) {
                val += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = val;
        }
    }
}

int main() {
    int sizes[] = { 100, 250, 500, 1000, 1500 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int idx = 0; idx < num_sizes; ++idx) {
        int size = sizes[idx];
        int matrix_size = size * size * sizeof(float);

        // Allocate memory for matrices on host
        float* h_A = (float*)malloc(matrix_size);
        float* h_B = (float*)malloc(matrix_size);
        float* h_C = (float*)malloc(matrix_size);
        float* h_C_CUDA = (float*)malloc(matrix_size);

        // Initialize matrices with random values
        for (int i = 0; i < size * size; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Allocate memory for matrices on device
        float* d_A, * d_B, * d_C;
        cudaMalloc((void**)&d_A, matrix_size);
        cudaMalloc((void**)&d_B, matrix_size);
        cudaMalloc((void**)&d_C, matrix_size);

        // Transfer data from host to device and measure time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Matrix size: %dx%d, Transfer time: %.2f ms\n", size, size, milliseconds);

        // Define execution configuration
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Launch kernel
        matrixMul << <numBlocks, threadsPerBlock >> > (d_C, d_A, d_B, size);

        // Transfer data back to host
        cudaMemcpy(h_C_CUDA, d_C, matrix_size, cudaMemcpyDeviceToHost);

        // Perform matrix multiplication on host for validation
        matrixMulHost(h_C, h_A, h_B, size);

        // Compare the results
        bool success = true;
        float epsilon = 1e-5;
        for (int i = 0; i < size * size; ++i) {
            if (abs(h_C[i] - h_C_CUDA[i]) > epsilon) {
                success = false;
                break;
            }
        }

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Free host memory
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_CUDA);

        if (success) {
            printf("Test PASSED\n");
        }
        else {
            printf("Test FAILED\n");
        }
    }

    return 0;
}


// save to csv Host to Dev

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float* C, const float* A, const float* B, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float val = 0.0f;
        for (int k = 0; k < size; ++k) {
            val += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = val;
    }
}

// Host function
void matrixMulHost(float* C, const float* A, const float* B, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float val = 0.0f;
            for (int k = 0; k < size; ++k) {
                val += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = val;
        }
    }
}

int main() {
    FILE* fp;
    fp = fopen("transfer_times.csv", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(fp, "Matrix Size,Transfer Time (ms)\n");

    int sizes[] = { 100, 250, 500, 1000, 1500 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int idx = 0; idx < num_sizes; ++idx) {
        int size = sizes[idx];
        int matrix_size = size * size * sizeof(float);

        // Allocate memory for matrices on host
        float* h_A = (float*)malloc(matrix_size);
        float* h_B = (float*)malloc(matrix_size);

        // Initialize matrices with random values
        for (int i = 0; i < size * size; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Allocate memory for matrices on device
        float* d_A, * d_B;
        cudaMalloc((void**)&d_A, matrix_size);
        cudaMalloc((void**)&d_B, matrix_size);

        // Transfer data from host to device and measure time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        fprintf(fp, "%d,%.2f\n", size, milliseconds);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);

        // Free host memory
        free(h_A);
        free(h_B);
    }

    fclose(fp);
    printf("Data saved to transfer_times.csv\n");

    return 0;
}

//Save to csv Dev to Host
// 
// CUDA kernel for matrix multiplication
__global__ void matrixMul(float* C, const float* A, const float* B, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float val = 0.0f;
        for (int k = 0; k < size; ++k) {
            val += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = val;
    }
}

// Host function
void matrixMulHost(float* C, const float* A, const float* B, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float val = 0.0f;
            for (int k = 0; k < size; ++k) {
                val += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = val;
        }a
    }
}

int main() {
    FILE* fp;
    fp = fopen("transfer_times_back.csv", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(fp, "Matrix Size,Transfer Time (ms)\n");

    int sizes[] = { 100, 250, 500, 1000, 1500 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int idx = 0; idx < num_sizes; ++idx) {
        int size = sizes[idx];
        int matrix_size = size * size * sizeof(float);

        // Allocate memory for matrices on host
        float* h_A = (float*)malloc(matrix_size);
        float* h_B = (float*)malloc(matrix_size);

        // Initialize matrices with random values
        for (int i = 0; i < size * size; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Allocate memory for matrices on device
        float* d_A, * d_B;
        cudaMalloc((void**)&d_A, matrix_size);
        cudaMalloc((void**)&d_B, matrix_size);

        // Transfer data from host to device and measure time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds_transfer_to_device = 0;
        cudaEventElapsedTime(&milliseconds_transfer_to_device, start, stop);

        // Transfer data back from device to host and measure time
        cudaEventRecord(start);
        cudaMemcpy(h_A, d_A, matrix_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B, d_B, matrix_size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds_transfer_to_host = 0;
        cudaEventElapsedTime(&milliseconds_transfer_to_host, start, stop);

        fprintf(fp, "%d,%.2f\n", size, milliseconds_transfer_to_host);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);

        // Free host memory
        free(h_A);
        free(h_B);
    }

    fclose(fp);
    printf("Data saved to transfer_times_back.csv\n");

    return 0;
}
//STEP 2

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float* C, const float* A, const float* B, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float val = 0.0f;
        for (int k = 0; k < size; ++k) {
            val += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = val;
    }
}

// Host function for matrix multiplication
void matrixMulHost(float* C, const float* A, const float* B, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float val = 0.0f;
            for (int k = 0; k < size; ++k) {
                val += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = val;
        }
    }
}

// Function to check if the GPU and CPU results match within a tolerance
bool checkResults(const float* A, const float* B, int size, float epsilon) {
    for (int i = 0; i < size * size; ++i) {
        if (abs(A[i] - B[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    FILE* fp;
    fp = fopen("multiplication_times32.csv", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(fp, "Matrix Size,CPU Time (ms),GPU Time (ms)\n");

    int sizes[] = { 100, 250, 500, 1000, 1500 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int idx = 0; idx < num_sizes; ++idx) {
        int size = sizes[idx];
        int matrix_size = size * size * sizeof(float);

        // Allocate memory for matrices on host
        float* h_A = (float*)malloc(matrix_size);
        float* h_B = (float*)malloc(matrix_size);
        float* h_C_CPU = (float*)malloc(matrix_size);
        float* h_C_GPU = (float*)malloc(matrix_size);

        // Initialize matrices with random values
        for (int i = 0; i < size * size; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Perform matrix multiplication on CPU and measure time
        cudaEvent_t start_CPU, stop_CPU;
        cudaEventCreate(&start_CPU);
        cudaEventCreate(&stop_CPU);

        cudaEventRecord(start_CPU);
        matrixMulHost(h_C_CPU, h_A, h_B, size);
        cudaEventRecord(stop_CPU);

        cudaEventSynchronize(stop_CPU);
        float milliseconds_CPU = 0;
        cudaEventElapsedTime(&milliseconds_CPU, start_CPU, stop_CPU);

        // Perform matrix multiplication on GPU and measure time
        cudaEvent_t start_GPU, stop_GPU;
        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);

        cudaEventRecord(start_GPU);

        float* d_A, * d_B, * d_C;
        cudaMalloc((void**)&d_A, matrix_size);
        cudaMalloc((void**)&d_B, matrix_size);
        cudaMalloc((void**)&d_C, matrix_size);

        cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(1, 1);
        dim3 numBlocks(1, 1);
        matrixMul << <numBlocks, threadsPerBlock >> > (d_C, d_A, d_B, size);

        cudaMemcpy(h_C_GPU, d_C, matrix_size, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        cudaEventRecord(stop_GPU);
        cudaEventSynchronize(stop_GPU);
        float milliseconds_GPU = 0;
        cudaEventElapsedTime(&milliseconds_GPU, start_GPU, stop_GPU);

        // Check if CPU and GPU results match
        float epsilon = 1e-5;
        bool match = checkResults(h_C_CPU, h_C_GPU, size, epsilon);

        fprintf(fp, "%d,%.2f,%.2f\n", size, milliseconds_CPU, milliseconds_GPU);

        // Free host memory
        free(h_A);
        free(h_B);
        free(h_C_CPU);
        free(h_C_GPU);
    }

    fclose(fp);
    printf("Data saved to multiplication_times.csv\n");

    return 0;
}

