#include <stdio.h>
#include "utils.h"
#include "secp256k1.h"

// Define structures needed by keccak256
struct _uint256 {
    uint32_t a, b, c, d, e, f, g, h;
};

#include "keccak256_mrspike.h"

#include "chacha_rng.h"

__device__ int eth_done_ultra = 0;
__device__ unsigned long long eth_count_ultra = 0;

__device__ inline void address_to_hex_ultra(const uint8_t *bytes, char *hex) {
    const char hex_chars[] = "0123456789abcdef";
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        hex[i * 2] = hex_chars[bytes[i] >> 4];
        hex[i * 2 + 1] = hex_chars[bytes[i] & 0xF];
    }
    hex[40] = '\0';
}

__device__ inline bool matches_target_eth_ultra(const uint8_t *address, const char *target, uint64_t target_len,
                                                 const char *suffix, uint64_t suffix_len) {
    const char *hex = "0123456789abcdef";
    for (uint64_t i = 0; i < target_len; ++i) {
        if (hex[(address[i/2] >> ((i & 1) ? 0 : 4)) & 15] != target[i]) return false;
    }
    for (uint64_t i = 0; i < suffix_len; ++i) {
        uint64_t pos = 40 - suffix_len + i;
        if (hex[(address[pos/2] >> ((pos & 1) ? 0 : 4)) & 15] != suffix[i]) return false;
    }
    return true;
}

// Ultra-optimized kernel with configurable parameters
__global__ void eth_vanity_search_ultra(uint8_t *buffer, uint64_t iterations_per_thread) {
    // Deconstruct buffer
    uint8_t *seed = buffer;
    uint64_t target_len;
    memcpy(&target_len, buffer + 32, 8);
    char *target = (char*)(buffer + 40);
    uint64_t suffix_len;
    memcpy(&suffix_len, buffer + 40 + target_len, 8);
    char *suffix = (char*)(buffer + 40 + target_len + 8);
    uint8_t *out = buffer + 40 + target_len + suffix_len + 8;

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char local_private[32];
    unsigned char local_public[65];
    unsigned char batch_public[SECP256K1_BATCH_SIZE * 65];
    char address_hex[41];

    ChaChaRng rng(seed, idx);
    // Reserve the top 1/256 of the scalar space: even a u64-length walk
    // stays below the curve order. Reject zero instead of modifying random bits.
    bool nonzero;
    do {
        rng.next32(local_private);
        nonzero = false;
        for (int i = 0; i < 32; ++i) nonzero |= local_private[i] != 0;
    } while (!nonzero || local_private[0] == 0xff);
    secp256k1_get_public_key(local_private, local_public);

    // Use provided iterations per thread
    for (uint64_t iter = 0; iter < iterations_per_thread; iter++) {
        // Check frequently to allow early exit when another thread finds a match
        if ((iter & 0x3F) == 0) {  // Check every 64 iterations
            if (atomicMax(&eth_done_ultra, 0) == 1) {
                atomicAdd(&eth_count_ultra, iter);
                return;
            }
        }

        // Extract x and y coordinates from public key (skip 0x04 prefix)
        _uint256 x, y;
        x.a = (local_public[1] << 24) | (local_public[2] << 16) | (local_public[3] << 8) | local_public[4];
        x.b = (local_public[5] << 24) | (local_public[6] << 16) | (local_public[7] << 8) | local_public[8];
        x.c = (local_public[9] << 24) | (local_public[10] << 16) | (local_public[11] << 8) | local_public[12];
        x.d = (local_public[13] << 24) | (local_public[14] << 16) | (local_public[15] << 8) | local_public[16];
        x.e = (local_public[17] << 24) | (local_public[18] << 16) | (local_public[19] << 8) | local_public[20];
        x.f = (local_public[21] << 24) | (local_public[22] << 16) | (local_public[23] << 8) | local_public[24];
        x.g = (local_public[25] << 24) | (local_public[26] << 16) | (local_public[27] << 8) | local_public[28];
        x.h = (local_public[29] << 24) | (local_public[30] << 16) | (local_public[31] << 8) | local_public[32];

        y.a = (local_public[33] << 24) | (local_public[34] << 16) | (local_public[35] << 8) | local_public[36];
        y.b = (local_public[37] << 24) | (local_public[38] << 16) | (local_public[39] << 8) | local_public[40];
        y.c = (local_public[41] << 24) | (local_public[42] << 16) | (local_public[43] << 8) | local_public[44];
        y.d = (local_public[45] << 24) | (local_public[46] << 16) | (local_public[47] << 8) | local_public[48];
        y.e = (local_public[49] << 24) | (local_public[50] << 16) | (local_public[51] << 8) | local_public[52];
        y.f = (local_public[53] << 24) | (local_public[54] << 16) | (local_public[55] << 8) | local_public[56];
        y.g = (local_public[57] << 24) | (local_public[58] << 16) | (local_public[59] << 8) | local_public[60];
        y.h = (local_public[61] << 24) | (local_public[62] << 16) | (local_public[63] << 8) | local_public[64];

        // Compute Ethereum address using MrSpike63's Keccak
        uint8_t address_20[20];
        keccak256_address(x, y, address_20);

        // Check if it matches target
        if (matches_target_eth_ultra(address_20, target, target_len, suffix, suffix_len)) {
            // Are we first to write result?
            if (atomicMax(&eth_done_ultra, 1) == 0) {
                address_to_hex_ultra(address_20, address_hex);
                // Copy private key (32 bytes), public key (65 bytes), and address hex (40 bytes)
                memcpy(out, local_private, 32);
                memcpy(out + 32, local_public, 65);
                memcpy(out + 97, address_hex, 40);
            }

            atomicAdd(&eth_count_ultra, iter + 1);
            return;
        }

        // Increment private key interpreted as big-endian (to match scalar parsing)
        unsigned int carry = 1;
        #pragma unroll
        for (int i = 31; i >= 0; i--) {
            unsigned int sum = (unsigned int)local_private[i] + carry;
            local_private[i] = (unsigned char)(sum & 0xFF);
            carry = (sum >> 8) & 0x1;
            if (!carry) break;
        }
        if ((iter & (SECP256K1_BATCH_SIZE - 1)) == 0) secp256k1_public_add_batch(local_public, batch_public);
        memcpy(local_public, batch_public + (iter & (SECP256K1_BATCH_SIZE - 1)) * 65, 65);
    }

    // Add final iteration count
    atomicAdd(&eth_count_ultra, iterations_per_thread);
}

// Persistent GPU context per device with configurable parameters
struct EthGpuContextUltra {
    uint8_t *d_buffer;
    size_t buffer_size;
    int num_blocks;
    int num_threads;
    uint64_t iterations_per_thread;
    bool initialized;
};

static EthGpuContextUltra gpu_contexts_ultra[16] = {{nullptr, 0, 0, 0, 0, false}};

extern "C" int eth_vanity_round_ultra(
    int id,
    uint8_t *seed,
    char *target,
    char *suffix,
    uint64_t target_len,
    uint64_t suffix_len,
    uint8_t *out,
    bool case_insensitive,
    int num_blocks,
    int num_threads,
    uint64_t iterations_per_thread)
{
    cudaError_t err;

    // Validate GPU ID
    if (id < 0 || id >= 16) {
        fprintf(stderr, "Invalid GPU ID: %d\n", id);
        return -1;
    }

    // Initialize GPU context if needed
    if (!gpu_contexts_ultra[id].initialized) {
        err = cudaSetDevice(id);
        if (err != cudaSuccess) { fprintf(stderr, "CUDA setDevice error: %s\n", cudaGetErrorString(err)); return -2; }

        // Allocate persistent buffer
        gpu_contexts_ultra[id].buffer_size = 32 + 8 + 256 + 8 + 256 + 137;
        err = cudaMalloc((void **)&gpu_contexts_ultra[id].d_buffer, gpu_contexts_ultra[id].buffer_size);
        if (err != cudaSuccess) { fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err)); return -3; }

        gpu_contexts_ultra[id].num_blocks = num_blocks;
        gpu_contexts_ultra[id].num_threads = num_threads;
        gpu_contexts_ultra[id].iterations_per_thread = iterations_per_thread;
        gpu_contexts_ultra[id].initialized = true;
    }

    err = cudaSetDevice(id);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA setDevice error: %s\n", cudaGetErrorString(err)); return -2; }

    // Copy data to device
    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer, seed, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (seed): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 32, &target_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (target_len): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 40, target, target_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (target): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 40 + target_len, &suffix_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (suffix_len): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 40 + target_len + 8, suffix, suffix_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (suffix): %s\n", cudaGetErrorString(err)); return -4; }

    // Reset done and count
    int zero = 0;
    unsigned long long zero_ull = 0;
    err = cudaMemcpyToSymbol(eth_done_ultra, &zero, sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpyToSymbol(eth_count_ultra, &zero_ull, sizeof(unsigned long long));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); return -4; }

    // Zero the output buffer on device
    uint8_t zeros[137] = {0};
    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 40 + target_len + suffix_len + 8, zeros, 137, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (zero output): %s\n", cudaGetErrorString(err)); return -4; }

    // Launch kernel with configured parameters
    eth_vanity_search_ultra<<<num_blocks, num_threads>>>(gpu_contexts_ultra[id].d_buffer, iterations_per_thread);

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA synchronize error: %s\n", cudaGetErrorString(err)); return -5; }

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err)); return -6; }

    // Copy result back
    err = cudaMemcpy(out, gpu_contexts_ultra[id].d_buffer + 40 + target_len + suffix_len + 8, 137, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (out): %s\n", cudaGetErrorString(err)); return -4; }

    // Copy count
    err = cudaMemcpyFromSymbol(out + 137, eth_count_ultra, 8, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); return -4; }

    // Copy done flag
    err = cudaMemcpyFromSymbol(out + 145, eth_done_ultra, 4, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); return -4; }

    return 0;
}

// Cleanup function
extern "C" void eth_vanity_cleanup_ultra(int id) {
    if (id >= 0 && id < 16 && gpu_contexts_ultra[id].initialized) {
        cudaSetDevice(id);
        cudaFree(gpu_contexts_ultra[id].d_buffer);
        gpu_contexts_ultra[id].d_buffer = nullptr;
        gpu_contexts_ultra[id].initialized = false;
    }
}
