#include <stdio.h>
#include "base58.h"
#include "utils.h"
#include "ed25519/ed25519.h"
#include "ed25519/ge.h"
#include "ed25519/sha512.h"

// XorShift128+ PRNG state
struct xorshift128plus_state {
    uint64_t s[2];
};

__device__ void init_xorshift_keypair(xorshift128plus_state &st,
                              const uint8_t *seed,
                              uint64_t idx)
{
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

__device__ uint64_t xorshift128plus_next_keypair(xorshift128plus_state &st) {
    uint64_t s1 = st.s[0], s0 = st.s[1];
    uint64_t result = s0 + s1;
    st.s[0] = s0;
    s1 ^= s1 << 23;
    st.s[1] = (s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5));
    return result;
}

__device__ int keypair_done = 0;
__device__ unsigned long long keypair_count = 0;
__device__ bool keypair_case_insensitive = false;

// Proper Ed25519 keypair generation using the solanity implementation
__device__ void ed25519_create_keypair_device(unsigned char *public_key, unsigned char *private_key, const unsigned char *seed) {
    ge_p3 A;

    sha512(seed, 32, private_key);
    private_key[0] &= 248;
    private_key[31] &= 63;
    private_key[31] |= 64;

    ge_scalarmult_base(&A, private_key);
    ge_p3_tobytes(public_key, &A);
}

__device__ bool matches_target_keypair(unsigned char *a, unsigned char *target, uint64_t n, unsigned char *suffix, uint64_t suffix_len, ulong encoded_len)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != target[i])
            return false;
    }
    for (int i = 0; i < suffix_len; i++)
    {
        if (a[encoded_len - suffix_len + i] != suffix[i])
            return false;
    }
    return true;
}

__global__ void
keypair_vanity_search(uint8_t *buffer, uint64_t stride)
{
    // Deconstruct buffer
    uint8_t *seed = buffer;
    uint64_t target_len;
    memcpy(&target_len, buffer + 32, 8);
    uint8_t *target = buffer + 40;
    uint64_t suffix_len;
    memcpy(&suffix_len, buffer + 40 + target_len, 8);
    uint8_t *suffix = buffer + 40 + target_len + 8;
    uint8_t *out = buffer + 40 + target_len + suffix_len + 8;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char local_seed[32];
    unsigned char local_private[64];
    unsigned char local_public[32];
    unsigned char local_encoded[44];

    // Initialize XorShift128+ state
    xorshift128plus_state st;
    init_xorshift_keypair(st, seed, idx);

    for (uint64_t iter = 0; iter < uint64_t(100) * 1000 * 1000; iter++)
    {
        // Check if someone found a result every 100 iterations
        if (iter % 100 == 0)
        {
            if (atomicMax(&keypair_done, 0) == 1)
            {
                atomicAdd(&keypair_count, iter);
                return;
            }
        }

        // Generate random 32-byte seed for Ed25519
        for (int i = 0; i < 4; ++i) {
            uint64_t rnd = xorshift128plus_next_keypair(st);
            memcpy(&local_seed[i * 8], &rnd, 8);
        }

        // Generate proper Ed25519 keypair from seed
        ed25519_create_keypair_device(local_public, local_private, local_seed);

        // Encode public key to base58 using simple, correct implementation
        ulong encoded_len = simple_base58_encode_32(local_public, local_encoded);

        // Check if it matches target
        if (matches_target_keypair(local_encoded, target, target_len, suffix, suffix_len, encoded_len))
        {
            // Are we first to write result?
            if (atomicMax(&keypair_done, 1) == 0)
            {
                // Copy seed (32 bytes), private key (64 bytes), public key (32 bytes), and encoded address (44 bytes)
                memcpy(out, local_seed, 32);
                memcpy(out + 32, local_private, 64);
                memcpy(out + 96, local_public, 32);
                memcpy(out + 128, local_encoded, 44);  // Copy the base58-encoded address
            }

            atomicAdd(&keypair_count, iter + 1);
            return;
        }
    }

    // Add final iteration count
    atomicAdd(&keypair_count, uint64_t(100) * 1000 * 1000);
}

// Static initialization flag per GPU
// States: -1 = failed, 0 = uninitialized, 1 = initializing, 2 = ready
static volatile int gpu_initialized[16] = {0};

extern "C" void keypair_vanity_round(
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
    if (gpu_initialized[id] == -1) {
        return; // GPU initialization previously failed permanently
    }

    // Only initialize GPU once per device
    if (gpu_initialized[id] == 0)
    {
        gpu_initialized[id] = 1; // Mark as initializing

        int deviceCount;
        err = cudaGetDeviceCount(&deviceCount);

        if (err != cudaSuccess)
        {
            printf("CUDA error getting device count: %s (code %d)\n", cudaGetErrorString(err), err);
            gpu_initialized[id] = -1; // Mark as permanently failed
            return;
        }

        if (deviceCount == 0)
        {
            printf("No CUDA devices found\n");
            gpu_initialized[id] = -1; // Mark as permanently failed
            return;
        }

        if (id >= deviceCount)
        {
            printf("Invalid GPU index: %d (only %d devices available)\n", id, deviceCount);
            gpu_initialized[id] = -1; // Mark as permanently failed
            return;
        }

        // Device and gpu_init are already called by cuda_init() from main thread
        gpu_initialized[id] = 2; // Mark as fully initialized
    }
    else if (gpu_initialized[id] == 1)
    {
        // Wait for initialization to complete
        while (gpu_initialized[id] == 1) {
            // Spin wait
        }

        // Check if initialization succeeded
        if (gpu_initialized[id] == -1) {
            return; // Initialization failed permanently
        }
    }
    else if (gpu_initialized[id] != 2)
    {
        return; // GPU not ready
    }

    // Allocate device buffer: seed (32) + target_len (8) + target + suffix_len (8) + suffix + output (172: 32 seed + 64 privkey + 32 pubkey + 44 address)
    uint8_t *d_buffer;
    size_t buffer_size = 32 + 8 + target_len + 8 + suffix_len + 172;
    err = cudaMalloc((void **)&d_buffer, buffer_size);
    if (err != cudaSuccess)
    {
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

    err = cudaMemcpyToSymbol(keypair_case_insensitive, &case_insensitive, 1, 0, cudaMemcpyHostToDevice);

    // Reset done and count
    int zero = 0;
    unsigned long long zero_ull = 0;
    err = cudaMemcpyToSymbol(keypair_done, &zero, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpyToSymbol(keypair_count, &zero_ull, sizeof(unsigned long long));
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    // Zero the output buffer on device
    uint8_t zeros[172] = {0};
    err = cudaMemcpy(d_buffer + 40 + target_len + suffix_len + 8, zeros, 172, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (zero output): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    // Configure kernel launch parameters
    const int num_threads = 256;  // Threads per block
    const int num_blocks = 1024;  // Number of blocks

    // Launch kernel
    keypair_vanity_search<<<num_blocks, num_threads>>>(d_buffer, num_blocks * num_threads);
    err = cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    // Copy result back (172 bytes: 32 seed + 64 private key + 32 public key + 44 address)
    err = cudaMemcpy(out, d_buffer + 40 + target_len + suffix_len + 8, 172, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (out): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpyFromSymbol(out + 172, keypair_count, 8, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    // Copy the "done" flag (4 bytes for int)
    err = cudaMemcpyFromSymbol(out + 180, keypair_done, 4, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    cudaFree(d_buffer);
}
