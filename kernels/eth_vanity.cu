#include <stdio.h>
#include "utils.h"
#include "secp256k1.h"

// Define structures needed by keccak256_mrspike.h
struct _uint256 {
    uint32_t a, b, c, d, e, f, g, h;
};

#include "keccak256_mrspike.h"

// XorShift128+ PRNG state
struct xorshift128plus_state {
    uint64_t s[2];
};

__device__ void init_xorshift_eth(xorshift128plus_state &st, const uint8_t *seed, uint64_t idx) {
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

__device__ uint64_t xorshift128plus_next_eth(xorshift128plus_state &st) {
    uint64_t s1 = st.s[0], s0 = st.s[1];
    uint64_t result = s0 + s1;
    st.s[0] = s0;
    s1 ^= s1 << 23;
    st.s[1] = (s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5));
    return result;
}

__device__ int eth_done = 0;
__device__ unsigned long long eth_count = 0;
// Note: Ethereum matching is always case-insensitive (patterns normalized to lowercase)

// Convert hex char to value
__device__ inline uint8_t hex_char_to_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

// Convert byte to hex chars
__device__ inline void byte_to_hex(uint8_t b, char *out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[b >> 4];
    out[1] = hex_chars[b & 0x0F];
}

// Convert address bytes to hex string (without 0x prefix)
__device__ void address_to_hex(const uint8_t *address_bytes, char *hex_str) {
    for (int i = 0; i < 20; i++) {
        byte_to_hex(address_bytes[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}

// Ethereum address matching - patterns are always lowercase, addresses are always lowercase
__device__ bool matches_target_eth(
    char *address_hex,
    char *target,
    uint64_t target_len,
    char *suffix,
    uint64_t suffix_len)
{
    // Check prefix
    for (int i = 0; i < target_len; i++) {
        if (address_hex[i] != target[i]) {
            return false;
        }
    }

    // Check suffix
    for (int i = 0; i < suffix_len; i++) {
        if (address_hex[40 - suffix_len + i] != suffix[i]) {
            return false;
        }
    }

    return true;
}

__global__ void eth_vanity_search(uint8_t *buffer, uint64_t stride) {
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
    unsigned char address_bytes[32];  // Keccak256 output
    char address_hex[41];

    // Initialize XorShift128+ state
    xorshift128plus_state st;
    init_xorshift_eth(st, seed, idx);

    for (uint64_t iter = 0; iter < uint64_t(20) * 1000 * 1000; iter++) {  // 20M iterations per thread
        // Check if someone found a result every 100 iterations
        if (iter % 100 == 0) {
            if (atomicMax(&eth_done, 0) == 1) {
                atomicAdd(&eth_count, iter);
                return;
            }
        }

        // Generate random 32-byte private key
        for (int i = 0; i < 4; ++i) {
            uint64_t rnd = xorshift128plus_next_eth(st);
            memcpy(&local_private[i * 8], &rnd, 8);
        }

        // Ensure private key is valid (non-zero and less than curve order)
        // For simplicity, just ensure it's non-zero
        bool all_zero = true;
        for (int i = 0; i < 32; i++) {
            if (local_private[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) continue;

        // Generate secp256k1 public key (uncompressed, 65 bytes)
        secp256k1_get_public_key(local_private, local_public);

        // Extract x and y coordinates from public key (skip 0x04 prefix)
        // Public key format: 0x04 || x (32 bytes) || y (32 bytes)
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
        address_to_hex(address_20, address_hex);

        // Check if it matches target
        if (matches_target_eth(address_hex, target, target_len, suffix, suffix_len)) {
            // Are we first to write result?
            if (atomicMax(&eth_done, 1) == 0) {
                // Copy private key (32 bytes), public key (65 bytes), and address hex (40 bytes)
                memcpy(out, local_private, 32);
                memcpy(out + 32, local_public, 65);
                memcpy(out + 97, address_hex, 40);
            }

            atomicAdd(&eth_count, iter + 1);
            return;
        }
    }

    // Add final iteration count
    atomicAdd(&eth_count, uint64_t(20) * 1000 * 1000);
}

// Static initialization flag per GPU
static volatile int gpu_initialized_eth[16] = {0};

extern "C" void eth_vanity_round(
    int id,
    uint8_t *seed,
    char *target,
    char *suffix,
    uint64_t target_len,
    uint64_t suffix_len,
    uint8_t *out,
    bool case_insensitive)
{
    cudaError_t err;

    // Check if GPU already failed initialization
    if (gpu_initialized_eth[id] == -1) {
        return;
    }

    // Only initialize GPU once per device
    if (gpu_initialized_eth[id] == 0) {
        gpu_initialized_eth[id] = 1;

        int deviceCount;
        err = cudaGetDeviceCount(&deviceCount);

        if (err != cudaSuccess) {
            printf("CUDA error getting device count: %s (code %d)\n", cudaGetErrorString(err), err);
            gpu_initialized_eth[id] = -1;
            return;
        }

        if (deviceCount == 0) {
            printf("No CUDA devices found\n");
            gpu_initialized_eth[id] = -1;
            return;
        }

        if (id >= deviceCount) {
            printf("Invalid GPU index: %d (only %d devices available)\n", id, deviceCount);
            gpu_initialized_eth[id] = -1;
            return;
        }

        gpu_initialized_eth[id] = 2;
    }
    else if (gpu_initialized_eth[id] == 1) {
        while (gpu_initialized_eth[id] == 1) {}
        if (gpu_initialized_eth[id] == -1) return;
    }
    else if (gpu_initialized_eth[id] != 2) {
        return;
    }

    // Allocate device buffer: seed (32) + target_len (8) + target + suffix_len (8) + suffix + output (137: 32 privkey + 65 pubkey + 40 address)
    uint8_t *d_buffer;
    size_t buffer_size = 32 + 8 + target_len + 8 + suffix_len + 137;
    err = cudaMalloc((void **)&d_buffer, buffer_size);
    if (err != cudaSuccess) {
        printf("CUDA malloc error: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy data to device
    err = cudaMemcpy(d_buffer, seed, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (seed): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpy(d_buffer + 32, &target_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (target_len): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpy(d_buffer + 40, target, target_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (target): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpy(d_buffer + 40 + target_len, &suffix_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (suffix_len): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpy(d_buffer + 40 + target_len + 8, suffix, suffix_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (suffix): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    // Note: case_insensitive parameter is ignored for Ethereum (always lowercase matching)

    // Reset done and count
    int zero = 0;
    unsigned long long zero_ull = 0;
    err = cudaMemcpyToSymbol(eth_done, &zero, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpyToSymbol(eth_count, &zero_ull, sizeof(unsigned long long));
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    // Zero the output buffer on device
    uint8_t zeros[137] = {0};
    err = cudaMemcpy(d_buffer + 40 + target_len + suffix_len + 8, zeros, 137, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (zero output): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    // Configure kernel launch parameters - Balance between parallelism and register pressure
    const int num_threads = 256;
    const int num_blocks = 2048;  // Moderate increase from 1024

    // Launch kernel
    eth_vanity_search<<<num_blocks, num_threads>>>(d_buffer, num_blocks * num_threads);
    err = cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    // Copy result back (137 bytes: 32 privkey + 65 pubkey + 40 address)
    err = cudaMemcpy(out, d_buffer + 40 + target_len + suffix_len + 8, 137, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (out): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpyFromSymbol(out + 137, eth_count, 8, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    // Copy the "done" flag (4 bytes for int)
    err = cudaMemcpyFromSymbol(out + 145, eth_done, 4, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    cudaFree(d_buffer);
}
