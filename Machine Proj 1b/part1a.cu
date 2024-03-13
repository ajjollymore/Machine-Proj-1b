#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int deviceID = 0; deviceID < deviceCount; ++deviceID) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceID);

        printf("\nDevice %d: %s\n", deviceID, deviceProp.name);
        printf("Clock rate: %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        printf("Number of streaming multiprocessors (SM): %d\n", deviceProp.multiProcessorCount);
        printf("Number of cores: %d\n", 3328);
        printf("Warp size: %d\n", deviceProp.warpSize);
        printf("Global memory: %.2f GB\n", static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024 * 1024));
        printf("Constant memory: %.2f KB\n", static_cast<float>(deviceProp.totalConstMem) / 1024);
        printf("Shared memory per block: %d bytes\n", deviceProp.sharedMemPerBlock);
        printf("Registers per block: %d\n", deviceProp.regsPerBlock);
        printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max dimensions of a block: [%d, %d, %d]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Max dimensions of a grid: [%d, %d, %d]\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }

    return 0;
}
