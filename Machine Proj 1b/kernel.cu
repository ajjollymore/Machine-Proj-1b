#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <cuda.h>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
/*
FOR TA/PROFESSOR

Uncomment each step/section as needed, the top comment of each block describes which part, sections are repeated for ease of use
*/
//STEP 1

// CUDA kernel for matrix multiplication
#define BLOCK_SIZE 16
__global__ void squareMatMul(float* P, float* N, float* M, int dimension_width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = row + col * dimension_width;
    for (int k = 0; k < dimension_width; k++) {
        int temp_idx1 = row + k * dimension_width;
        int temp_idx2 = k + col * dimension_width;
        if (row < dimension_width && col < dimension_width) {
            P[idx] += N[temp_idx1] * M[temp_idx2];
        }
    }
}

//Host 
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

        float* h_A = (float*)malloc(matrix_size);
        float* h_B = (float*)malloc(matrix_size);
        float* h_C = (float*)malloc(matrix_size);
        float* h_C_CUDA = (float*)malloc(matrix_size);

        for (int i = 0; i < size * size; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        float* d_A, * d_B, * d_C;
        cudaMalloc((void**)&d_A, matrix_size);
        cudaMalloc((void**)&d_B, matrix_size);
        cudaMalloc((void**)&d_C, matrix_size);

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
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        squareMatMul << <numBlocks, threadsPerBlock >> > (d_C, d_A, d_B, size);
        cudaMemcpy(h_C_CUDA, d_C, matrix_size, cudaMemcpyDeviceToHost);
        matrixMulHost(h_C, h_A, h_B, size);
        bool success = true;
        float epsilon = 1e-5;
        for (int i = 0; i < size * size; ++i) {
            if (abs(h_C[i] - h_C_CUDA[i]) > epsilon) {
                success = false;
                break;
            }
        }
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

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
/*
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
*/
//Save to csv Dev to Host
/*
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
}*/
//STEP 2
/*
__global__ void Mul_NM(float* P, const float* N, const float* M, int dimension_width) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int row = 0; row < dimension_width; row++) {
            for (int col = 0; col < dimension_width; col++) {
                float sum = 0.0f;
                for (int k = 0; k < dimension_width; k++) {
                    sum += N[row * dimension_width + k] * M[k * dimension_width + col];
                }
                P[row * dimension_width + col] = sum;
            }
        }
    }
}


void MulHost(float* P, const float* N, const float* M, int dimension_width) {
    for (int x = 0; x < dimension_width; x++) {
        for (int y = 0; y < dimension_width; y++) {
            float val = 0.0f;
            for (int w = 0; w < dimension_width; w++) {
                int move_1 = x + dimension_width * w;
                int move_2 = w * dimension_width + y;
                val = val + N[move_1] * M[move_2];
            }
            P[x * dimension_width + y] = val;
        }
    }
}

int main(int argc, char* argv[]) {

    FILE* fp;
    fp = fopen("matrix32_times.csv", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(fp, "Matrix Size,Mul Time CPU (ms),Mul Time GPU (ms)\n");

    int dimension[5] = { 100, 250, 500, 1000, 1500 };
    for (int index_value = 0; index_value < 5; ++index_value)
    {
        int dimension_width = dimension[index_value];
        int matrix_dim = dimension_width * dimension_width * sizeof(float);

        float gpu_time = 0.0f;
        float cpu_time = 0.0f;

        float* hostN = (float*)malloc(matrix_dim);
        float* hostM = (float*)malloc(matrix_dim);
        float* hostP = (float*)malloc(matrix_dim);

        for (int i = 0; i < dimension_width * dimension_width; ++i) {
            hostN[i] = static_cast<float>(rand() / RAND_MAX);
            hostM[i] = static_cast<float>(rand() / RAND_MAX);
        }

        float* deviceN, * deviceM, * deviceP;

        cudaMalloc((void**)&deviceN, matrix_dim);
        cudaMalloc((void**)&deviceM, matrix_dim);
        cudaMalloc((void**)&deviceP, matrix_dim);

        cudaMemcpy(deviceN, hostN, matrix_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceM, hostM, matrix_dim, cudaMemcpyHostToDevice);

        cudaEvent_t start_instance, stop_instance;
        cudaEventCreate(&start_instance);
        cudaEventCreate(&stop_instance);
        cudaEventRecord(start_instance, 0);

        dim3 threadsPerBlock(1);
        dim3 numberOfBlocks(1);

        Mul_NM << < numberOfBlocks, threadsPerBlock >> > (deviceN, deviceM, deviceP, dimension_width);

        cudaEventRecord(stop_instance, 0);
        cudaEventSynchronize(stop_instance);
        cudaEventElapsedTime(&gpu_time, start_instance, stop_instance);

        cudaMemcpy(hostP, deviceP, matrix_dim, cudaMemcpyDeviceToHost);

        clock_t start = clock();
        MulHost(hostN, hostM, hostP, dimension_width);
        clock_t stop = clock();
        cpu_time = (float)(stop - start) * 1000.0f / CLOCKS_PER_SEC;

        cout << "GPU Time: " << gpu_time << endl;
        cout << "CPU Time: " << cpu_time << endl;

        fprintf(fp, "%d,%.2f,%.2f\n", dimension_width, cpu_time, gpu_time);

        cudaFree(deviceN);
        cudaFree(deviceM);
        cudaFree(deviceP);

        free(hostN);
        free(hostM);
        free(hostP);

        cudaEventDestroy(start_instance);
        cudaEventDestroy(stop_instance);
    }

    fclose(fp);
    printf("Data saved to matrix32_times.csv\n");

    return 0;
}
*/
//Step 3
 /*
__global__ void Mul_NM(float* P, float* N, float* M, int dimension_width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = row + col * dimension_width;
    for (int k = 0; k < dimension_width; k++) {
        int temp_idx1 = row + k * dimension_width;
        int temp_idx2 = k + col * dimension_width;
        if (row < dimension_width && col < dimension_width) {
            P[idx] += N[temp_idx1] * M[temp_idx2];
        }
    }
}

void MulHost(float* P, const float* N, const float* M, int dimension_width) {
    for (int x = 0; x < dimension_width; x++) {
        for (int y = 0; y < dimension_width; y++) {
            float val = 0.0f;
            for (int w = 0; w < dimension_width; w++) {
                int move_1 = x + dimension_width * w;
                int move_2 = w * dimension_width + y;
                val = val + N[move_1] * M[move_2];
            }
            P[x * dimension_width + y] = val;
        }
    }
}

int main() {
    FILE* fp;
    fp = fopen("matrix_P3_times.csv", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(fp, "Matrix Size,Block Size,Mul Time CPU (ms),Mul Time GPU (ms)\n");

    int dimension[5] = { 100, 250, 500, 1000, 1500 };
    int blocksize[5] = { 2, 5, 10, 25, 32 };

    for (int i = 0; i < 5; i++)
    {
        int blockdim = blocksize[i];

        for (int x = 0; x < 5; x++) {
            int dimension_width = dimension[x];
            size_t matrix_dim = dimension_width * dimension_width * sizeof(float);

            float gpu_time = 0.0f;
            float cpu_time = 0.0f;

            float* hostN = (float*)malloc(matrix_dim);
            float* hostM = (float*)malloc(matrix_dim);
            float* hostP = (float*)malloc(matrix_dim);

            for (int i = 0; i < dimension_width * dimension_width; i++) {
                hostN[i] = static_cast<float>(rand() / RAND_MAX);
                hostM[i] = static_cast<float>(rand() / RAND_MAX);
                hostP[i] = 0.0f;
            }

            float* deviceN, * deviceM, * deviceP;

            cudaMalloc((void**)&deviceN, matrix_dim);
            cudaMalloc((void**)&deviceM, matrix_dim);
            cudaMalloc((void**)&deviceP, matrix_dim);

            cudaMemcpy(deviceN, hostN, matrix_dim, cudaMemcpyHostToDevice);
            cudaMemcpy(deviceM, hostM, matrix_dim, cudaMemcpyHostToDevice);
            cudaMemcpy(deviceP, hostP, matrix_dim, cudaMemcpyHostToDevice);

            cudaEvent_t start_instance, stop_instance;
            cudaEventCreate(&start_instance);
            cudaEventCreate(&stop_instance);
            cudaEventRecord(start_instance, 0);

            dim3 threadsPerBlock(blockdim, blockdim);
            dim3 Blocks((int)ceil(dimension_width / (float)threadsPerBlock.x), (int)ceil(dimension_width / (float)threadsPerBlock.y));
            Mul_NM << <Blocks, threadsPerBlock >> > (deviceP, deviceN, deviceM, dimension_width);

            cudaEventRecord(stop_instance);
            cudaEventSynchronize(stop_instance);
            cudaEventElapsedTime(&gpu_time, start_instance, stop_instance);
            cudaEventDestroy(start_instance);
            cudaEventDestroy(stop_instance);

            cudaMemcpy(hostP, deviceP, matrix_dim, cudaMemcpyDeviceToHost);

            float* PTemp = (float*)malloc(matrix_dim);
            cudaEventCreate(&start_instance);
            cudaEventCreate(&stop_instance);
            cudaEventRecord(start_instance);

            MulHost(hostN, hostM, hostP, dimension_width);

            cudaEventRecord(stop_instance);
            cudaEventSynchronize(stop_instance);
            cudaEventElapsedTime(&cpu_time, start_instance, stop_instance);
            cudaEventDestroy(start_instance);
            cudaEventDestroy(stop_instance);

            cout << "Matrix Size: " << dimension_width << endl;
            cout << "Block Dim: " << blockdim << endl;
            cout << "GPU Time: " << gpu_time << endl;
            cout << "CPU Time: " << cpu_time << endl;
            cout << endl;

            fprintf(fp, "%d,%d,%.2f,%.2f\n", dimension_width, blockdim, cpu_time, gpu_time);

            cudaFree(deviceN);
            cudaFree(deviceM);
            cudaFree(deviceP);

            free(hostN);
            free(hostM);
            free(hostP);
        }
    }

    fclose(fp);
    printf("Data saved to matrix_P3_times.csv\n");

    return 0;
}
*/