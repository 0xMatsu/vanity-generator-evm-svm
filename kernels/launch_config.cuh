#pragma once
#include <cuda_runtime.h>

// Device limits alone are insufficient: registers and static shared memory
// can make a legal block size impossible for this particular compiled kernel.
template <typename Kernel>
cudaError_t vanity_launch_threads(Kernel kernel, int requested, int* selected) {
    if (requested < 1) return cudaErrorInvalidConfiguration;
    cudaFuncAttributes attributes;
    cudaError_t err = cudaFuncGetAttributes(&attributes, kernel);
    if (err != cudaSuccess) return err;
    int threads = requested;
    while (threads > 0) {
        if (threads <= attributes.maxThreadsPerBlock) {
            int active = 0;
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, 0);
            if (err != cudaSuccess) return err;
            if (active > 0) {
                *selected = threads;
                return cudaSuccess;
            }
        }
        threads /= 2;
    }
    return cudaErrorLaunchOutOfResources;
}
