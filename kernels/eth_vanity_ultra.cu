#include <stdio.h>
#include "utils.h"
#include "secp256k1.h"

// Define structures needed by keccak256
struct _uint256 {
    uint32_t a, b, c, d, e, f, g, h;
};

#include "keccak256_mrspike.h"

// XorShift128+ PRNG state
struct xorshift128plus_state {
    uint64_t s[2];
};

__device__ void init_xorshift_eth_ultra(xorshift128plus_state &st, const uint8_t *seed, uint64_t idx) {
    uint64_t k0 = *((const uint64_t*)(seed + 0));
    uint64_t k1 = *((const uint64_t*)(seed + 8));
    uint64_t k2 = *((const uint64_t*)(seed + 16));
    uint64_t k3 = *((const uint64_t*)(seed + 24));

    uint64_t z0 = k0 ^ k2;
    z0 += idx;
    z0 = (z0 ^ (z0 >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z0 = (z0 ^ (z0 >> 27)) * 0x94d049bb133111ebULL;
    st.s[0] = z0 ^ (z0 >> 31);

    uint64_t z1 = k1 ^ k3;
    z1 += idx + 0x9e3779b97f4a7c15ULL;
    z1 = (z1 ^ (z1 >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z1 = (z1 ^ (z1 >> 27)) * 0x94d049bb133111ebULL;
    st.s[1] = z1 ^ (z1 >> 31);
}

__device__ uint64_t xorshift128plus_next_eth_ultra(xorshift128plus_state &st) {
    uint64_t s1 = st.s[0], s0 = st.s[1];
    uint64_t result = s0 + s1;
    st.s[0] = s0;
    s1 ^= s1 << 23;
    st.s[1] = (s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5));
    return result;
}

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

__device__ inline bool matches_target_eth_ultra(const char *address, const char *target, uint64_t target_len,
                                                 const char *suffix, uint64_t suffix_len) {
    // Check prefix
    #pragma unroll 8
    for (uint64_t i = 0; i < target_len; i++) {
        if (address[i] != target[i]) return false;
    }
    // Check suffix
    #pragma unroll 8
    for (uint64_t i = 0; i < suffix_len; i++) {
        if (address[40 - suffix_len + i] != suffix[i]) return false;
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

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char local_private[32];
    unsigned char local_public[65];
    char address_hex[41];

    // Initialize XorShift128+ state
    xorshift128plus_state st;
    init_xorshift_eth_ultra(st, seed, idx);

    // Initialize first private key from PRNG once (non-zero) and its public key
    uint64_t rnd0 = xorshift128plus_next_eth_ultra(st);
    uint64_t rnd1 = xorshift128plus_next_eth_ultra(st);
    uint64_t rnd2 = xorshift128plus_next_eth_ultra(st);
    uint64_t rnd3 = xorshift128plus_next_eth_ultra(st);
    if ((rnd0 | rnd1 | rnd2 | rnd3) == 0) { rnd0 = 1; }
    memcpy(&local_private[0], &rnd0, 8);
    memcpy(&local_private[8], &rnd1, 8);
    memcpy(&local_private[16], &rnd2, 8);
    memcpy(&local_private[24], &rnd3, 8);
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

        // Convert to hex
        address_to_hex_ultra(address_20, address_hex);

        // Check if it matches target
        if (matches_target_eth_ultra(address_hex, target, target_len, suffix, suffix_len)) {
            // Are we first to write result?
            if (atomicMax(&eth_done_ultra, 1) == 0) {
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
        secp256k1_public_add_generator(local_public, local_public);
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
        printf("Invalid GPU ID: %d\n", id);
        return -1;
    }

    // Initialize GPU context if needed
    if (!gpu_contexts_ultra[id].initialized) {
        err = cudaSetDevice(id);
        if (err != cudaSuccess) { printf("CUDA setDevice error: %s\n", cudaGetErrorString(err)); return -2; }

        // Allocate persistent buffer
        gpu_contexts_ultra[id].buffer_size = 32 + 8 + 256 + 8 + 256 + 137;
        err = cudaMalloc((void **)&gpu_contexts_ultra[id].d_buffer, gpu_contexts_ultra[id].buffer_size);
        if (err != cudaSuccess) { printf("CUDA malloc error: %s\n", cudaGetErrorString(err)); return -3; }

        gpu_contexts_ultra[id].num_blocks = num_blocks;
        gpu_contexts_ultra[id].num_threads = num_threads;
        gpu_contexts_ultra[id].iterations_per_thread = iterations_per_thread;
        gpu_contexts_ultra[id].initialized = true;
    }

    err = cudaSetDevice(id);
    if (err != cudaSuccess) { printf("CUDA setDevice error: %s\n", cudaGetErrorString(err)); return -2; }

    // Copy data to device
    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer, seed, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (seed): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 32, &target_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (target_len): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 40, target, target_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (target): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 40 + target_len, &suffix_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (suffix_len): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 40 + target_len + 8, suffix, suffix_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (suffix): %s\n", cudaGetErrorString(err)); return -4; }

    // Reset done and count
    int zero = 0;
    unsigned long long zero_ull = 0;
    err = cudaMemcpyToSymbol(eth_done_ultra, &zero, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpyToSymbol(eth_count_ultra, &zero_ull, sizeof(unsigned long long));
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); return -4; }

    // Zero the output buffer on device
    uint8_t zeros[137] = {0};
    err = cudaMemcpy(gpu_contexts_ultra[id].d_buffer + 40 + target_len + suffix_len + 8, zeros, 137, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (zero output): %s\n", cudaGetErrorString(err)); return -4; }

    // Launch kernel with configured parameters
    eth_vanity_search_ultra<<<num_blocks, num_threads>>>(gpu_contexts_ultra[id].d_buffer, iterations_per_thread);

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA synchronize error: %s\n", cudaGetErrorString(err)); return -5; }

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA launch error: %s\n", cudaGetErrorString(err)); return -6; }

    // Copy result back
    err = cudaMemcpy(out, gpu_contexts_ultra[id].d_buffer + 40 + target_len + suffix_len + 8, 137, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (out): %s\n", cudaGetErrorString(err)); return -4; }

    // Copy count
    err = cudaMemcpyFromSymbol(out + 137, eth_count_ultra, 8, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); return -4; }

    // Copy done flag
    err = cudaMemcpyFromSymbol(out + 145, eth_done_ultra, 4, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); return -4; }

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
