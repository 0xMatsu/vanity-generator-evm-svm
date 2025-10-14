#include <cuda_runtime.h>
#include <stdio.h>

// Forward declaration
extern "C" void gpu_init(int id);

extern "C" int cuda_init() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        // Only print errors to stderr
        fprintf(stderr, "Error: CUDA initialization failed: %s (code %d)\n", cudaGetErrorString(err), err);
        fflush(stderr);
        return (int)err;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA devices available\n");
        fflush(stderr);
        return -1;
    }

    // Set device flags BEFORE creating context (must be first CUDA call for the device)
    err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        // Continue anyway - not critical
    }

    // Initialize device 0
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to set CUDA device 0: %s (code %d)\n", cudaGetErrorString(err), err);
        fflush(stderr);
        return (int)err;
    }

    // Force CUDA context creation by allocating and freeing a small buffer
    void* dummy;
    err = cudaMalloc(&dummy, 1);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to allocate CUDA memory: %s (code %d)\n", cudaGetErrorString(err), err);
        fflush(stderr);
        return (int)err;
    }
    cudaFree(dummy);

    // Call gpu_init to set up global variables for kernel configuration
    gpu_init(0);

    return 0;
}
