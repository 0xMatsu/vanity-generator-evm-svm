#include <cuda_runtime.h>
#include <stdio.h>

// GPU configuration and auto-tuning

struct GpuConfig {
    int device_id;
    int sm_count;
    int max_threads_per_sm;
    int max_threads_per_block;
    int max_blocks_per_sm;
    size_t total_global_mem;
    size_t shared_mem_per_block;
    int compute_capability_major;
    int compute_capability_minor;

    // Calculated optimal values
    int optimal_threads_per_block;
    int optimal_blocks;
    int optimal_iterations_per_thread;
};

extern "C" int get_gpu_config(int device_id, GpuConfig *config) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return -1;
    }

    config->device_id = device_id;
    config->sm_count = prop.multiProcessorCount;
    config->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    config->max_threads_per_block = prop.maxThreadsPerBlock;
    config->max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    config->total_global_mem = prop.totalGlobalMem;
    config->shared_mem_per_block = prop.sharedMemPerBlock;
    config->compute_capability_major = prop.major;
    config->compute_capability_minor = prop.minor;

    // Calculate optimal configuration
    // For Ethereum (heavy computation), use all available resources
    config->optimal_threads_per_block = 256;  // Good balance for most GPUs

    // Target: saturate all SMs with multiple blocks per SM
    int blocks_per_sm = 4;  // Run 4 blocks per SM for good occupancy
    config->optimal_blocks = config->sm_count * blocks_per_sm;

    // Adjust iterations based on compute capability
    // Use smaller batches for better responsiveness and progress reporting
    if (config->compute_capability_major >= 8) {
        // Ampere or newer (RTX 30xx, 40xx, A100, etc.)
        config->optimal_iterations_per_thread = 10 * 1000;  // 10K iterations per thread
    } else if (config->compute_capability_major >= 7) {
        // Turing/Volta (RTX 20xx, V100, etc.)
        config->optimal_iterations_per_thread = 8 * 1000;  // 8K
    } else {
        // Pascal or older
        config->optimal_iterations_per_thread = 5 * 1000;  // 5K
    }

    return 0;
}

extern "C" void print_gpu_config(const GpuConfig *config) {
    printf("=== GPU Configuration ===\n");
    printf("Device ID: %d\n", config->device_id);
    printf("SM Count: %d\n", config->sm_count);
    printf("Compute Capability: %d.%d\n", config->compute_capability_major, config->compute_capability_minor);
    printf("Max Threads per SM: %d\n", config->max_threads_per_sm);
    printf("Max Threads per Block: %d\n", config->max_threads_per_block);
    printf("Total Global Memory: %.2f GB\n", config->total_global_mem / (1024.0 * 1024.0 * 1024.0));
    printf("\n=== Optimal Configuration ===\n");
    printf("Threads per Block: %d\n", config->optimal_threads_per_block);
    printf("Number of Blocks: %d\n", config->optimal_blocks);
    printf("Iterations per Thread: %dM\n", config->optimal_iterations_per_thread / (1000 * 1000));
    printf("Total Parallel Keys: %.2fB per batch\n",
           (double)config->optimal_blocks * config->optimal_threads_per_block *
           config->optimal_iterations_per_thread / (1000.0 * 1000.0 * 1000.0));
    printf("========================\n");
}
