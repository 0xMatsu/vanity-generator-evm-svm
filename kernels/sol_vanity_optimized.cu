#include <stdio.h>
#include "base58.h"
#include "vanity.h"
#include "sha256.h"
#include "ed25519/ed25519.h"
#include "ed25519/ge.h"
#include "ed25519/sha512.h"

// XorShift128+ PRNG state
struct xorshift128plus_state {
    uint64_t s[2];
};

__device__ void init_xorshift_sol_opt(xorshift128plus_state &st, const uint8_t *seed, uint64_t idx) {
    uint64_t k0 = *((const uint64_t*)(seed + 0));
    uint64_t k1 = *((const uint64_t*)(seed + 8));
    uint64_t k2 = *((const uint64_t*)(seed + 16));
    uint64_t k3 = *((const uint64_t*)(seed + 24));

    // Use SplitMix64 algorithm for better mixing of thread index with seed
    // This ensures independent PRNG streams even for adjacent threads
    uint64_t z0 = k0 ^ k2 ^ idx;
    z0 = (z0 ^ (z0 >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z0 = (z0 ^ (z0 >> 27)) * 0x94d049bb133111ebULL;
    z0 = (z0 ^ (z0 >> 31)) * 0x9e3779b97f4a7c15ULL;
    st.s[0] = z0;

    uint64_t z1 = k1 ^ k3 ^ (idx * 0x9e3779b97f4a7c15ULL);
    z1 = (z1 ^ (z1 >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z1 = (z1 ^ (z1 >> 27)) * 0x94d049bb133111ebULL;
    z1 = (z1 ^ (z1 >> 31)) * 0x9e3779b97f4a7c15ULL;
    st.s[1] = z1;
}

__device__ uint64_t xorshift128plus_next_sol_opt(xorshift128plus_state &st) {
    uint64_t s1 = st.s[0], s0 = st.s[1];
    uint64_t result = s0 + s1;
    st.s[0] = s0;
    s1 ^= s1 << 23;
    st.s[1] = (s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5));
    return result;
}

__device__ int sol_done_opt = 0;
__device__ unsigned long long sol_count_opt = 0;
__device__ unsigned long long sol_fullencode_count_opt = 0;
__device__ bool sol_case_insensitive_opt = false;

// Ed25519 keypair generation device function
__device__ void ed25519_create_keypair_device_sol(unsigned char *public_key, unsigned char *private_key, const unsigned char *seed) {
    ge_p3 A;

    sha512(seed, 32, private_key);
    private_key[0] &= 248;
    private_key[31] &= 63;
    private_key[31] |= 64;

    ge_scalarmult_base(&A, private_key);
    ge_p3_tobytes(public_key, &A);
}

__device__ bool matches_target_sol_opt(const char *address, uint64_t address_len, const char *target, uint64_t target_len,
                                       const char *suffix, uint64_t suffix_len, bool case_insensitive) {
    if (case_insensitive) {
        // Case-insensitive comparison
        for (uint64_t i = 0; i < target_len; i++) {
            char a = address[i];
            char t = target[i];
            // Convert to lowercase
            if (a >= 'A' && a <= 'Z') a = a + 32;
            if (t >= 'A' && t <= 'Z') t = t + 32;
            if (a != t) return false;
        }
        for (uint64_t i = 0; i < suffix_len; i++) {
            char a = address[address_len - suffix_len + i];
            char s = suffix[i];
            if (a >= 'A' && a <= 'Z') a = a + 32;
            if (s >= 'A' && s <= 'Z') s = s + 32;
            if (a != s) return false;
        }
    } else {
        // Case-sensitive comparison
        for (uint64_t i = 0; i < target_len; i++) {
            if (address[i] != target[i]) return false;
        }
        for (uint64_t i = 0; i < suffix_len; i++) {
            if (address[address_len - suffix_len + i] != suffix[i]) return false;
        }
    }
    return true;
}

// Optimized Solana vanity search kernel
__global__ void sol_vanity_search_optimized(uint8_t *buffer, uint64_t iterations_per_thread) {
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

    unsigned char local_seed[32];
    unsigned char private_key[64];
    unsigned char public_key[32];
    char address[45];
    char suffix_buf[16]; // supports quick checks up to 16-char suffix

    // Initialize XorShift128+ state
    xorshift128plus_state st;
    init_xorshift_sol_opt(st, seed, idx);

    // Seed-based per-iteration path (faster on Ed25519):

    for (uint64_t iter = 0; iter < iterations_per_thread; iter++) {
        // Check if someone found a result periodically
        if ((iter & 0x3F) == 0) { // every 64 iterations
            if (atomicMax(&sol_done_opt, 0) == 1) {
                atomicAdd(&sol_count_opt, iter);
                return;
            }
        }

        // Generate random 32-byte seed and derive keypair
        for (int i = 0; i < 4; ++i) {
            uint64_t rnd = xorshift128plus_next_sol_opt(st);
            memcpy(&local_seed[i * 8], &rnd, 8);
        }
        ed25519_create_keypair_device_sol(public_key, private_key, local_seed);

        // If suffix is requested, check it first using suffix-only encoding (fast reject)
        if (suffix_len > 0) {
            if (suffix_len <= 10) {
                if (!base58_suffix_match_mod_32(public_key, suffix, (int)suffix_len, sol_case_insensitive_opt)) {
                    continue;
                }
            } else {
                int got = simple_base58_suffix_32(public_key, suffix_buf, (int)(suffix_len > 15 ? 15 : suffix_len));
                bool suffix_ok = true;
                for (int i = 0; i < got; i++) {
                    char a = suffix_buf[i];
                    char b = suffix[suffix_len - 1 - i]; // compare from end
                    if (sol_case_insensitive_opt) {
                        if (a >= 'A' && a <= 'Z') a = a + 32;
                        if (b >= 'A' && b <= 'Z') b = b + 32;
                    }
                    if (a != b) { suffix_ok = false; break; }
                }
                if (!suffix_ok) {
                    continue;
                }
            }
        }

        // Encode full address only after passing suffix check (or if no suffix)
        unsigned char encoded[45];
        atomicAdd(&sol_fullencode_count_opt, 1ULL);
        ulong encoded_len = simple_base58_encode_32(public_key, encoded);
        for (ulong i = 0; i < encoded_len && i < 44; i++) { address[i] = encoded[i]; }
        address[encoded_len < 44 ? encoded_len : 44] = '\0';

        if (matches_target_sol_opt(address, encoded_len, target, target_len, suffix, suffix_len, sol_case_insensitive_opt)) {
            // Are we first to write result?
            if (atomicMax(&sol_done_opt, 1) == 0) {
                // Copy seed (32), private (64), pub (32), address (44) — wallet-compatible
                memcpy(out, local_seed, 32);
                memcpy(out + 32, private_key, 64);
                memcpy(out + 96, public_key, 32);
                memcpy(out + 128, address, 44);
            }

            atomicAdd(&sol_count_opt, iter + 1);
            return;
        }

        // Next iteration — new random seed
    }

    // Add final iteration count
    atomicAdd(&sol_count_opt, iterations_per_thread);
}

// Persistent GPU context per device
struct SolGpuContext {
    uint8_t *d_buffer;
    size_t buffer_size;
    bool initialized;
};

static SolGpuContext sol_gpu_contexts[16] = {{nullptr, 0, false}};

extern "C" int sol_vanity_round_optimized(
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
    if (!sol_gpu_contexts[id].initialized) {
        err = cudaSetDevice(id);
        if (err != cudaSuccess) { printf("CUDA setDevice error: %s\n", cudaGetErrorString(err)); return -2; }

        // Allocate persistent buffer
        sol_gpu_contexts[id].buffer_size = 32 + 8 + 256 + 8 + 256 + 172;  // seed + target_len + max_target + suffix_len + max_suffix + output
        err = cudaMalloc((void **)&sol_gpu_contexts[id].d_buffer, sol_gpu_contexts[id].buffer_size);
        if (err != cudaSuccess) { printf("CUDA malloc error: %s\n", cudaGetErrorString(err)); return -3; }

        sol_gpu_contexts[id].initialized = true;
    }

    err = cudaSetDevice(id);
    if (err != cudaSuccess) { printf("CUDA setDevice error: %s\n", cudaGetErrorString(err)); return -2; }

    // Copy data to device
    err = cudaMemcpy(sol_gpu_contexts[id].d_buffer, seed, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (seed): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(sol_gpu_contexts[id].d_buffer + 32, &target_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (target_len): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(sol_gpu_contexts[id].d_buffer + 40, target, target_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (target): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(sol_gpu_contexts[id].d_buffer + 40 + target_len, &suffix_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (suffix_len): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpy(sol_gpu_contexts[id].d_buffer + 40 + target_len + 8, suffix, suffix_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (suffix): %s\n", cudaGetErrorString(err)); return -4; }

    // Set case insensitive flag
    err = cudaMemcpyToSymbol(sol_case_insensitive_opt, &case_insensitive, sizeof(bool));
    if (err != cudaSuccess) { printf("CUDA memcpy error (case_insensitive): %s\n", cudaGetErrorString(err)); return -4; }

    // Reset done and count
    int zero = 0;
    unsigned long long zero_ull = 0;
    err = cudaMemcpyToSymbol(sol_done_opt, &zero, sizeof(int));
    if (err != cudaSuccess) { printf("CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); return -4; }

    err = cudaMemcpyToSymbol(sol_count_opt, &zero_ull, sizeof(unsigned long long));
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); return -4; }

    // Zero the output buffer on device
    uint8_t zeros[172] = {0};
    err = cudaMemcpy(sol_gpu_contexts[id].d_buffer + 40 + target_len + suffix_len + 8, zeros, 172, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (zero output): %s\n", cudaGetErrorString(err)); return -4; }

    // Launch kernel
    sol_vanity_search_optimized<<<num_blocks, num_threads>>>(sol_gpu_contexts[id].d_buffer, iterations_per_thread);

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA synchronize error: %s\n", cudaGetErrorString(err)); return -5; }

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA launch error: %s\n", cudaGetErrorString(err)); return -6; }

    // Copy result back (172 bytes: 32 seed + 64 private key + 32 public key + 44 address)
    err = cudaMemcpy(out, sol_gpu_contexts[id].d_buffer + 40 + target_len + suffix_len + 8, 172, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (out): %s\n", cudaGetErrorString(err)); return -4; }

    // Copy counts
    err = cudaMemcpyFromSymbol(out + 172, sol_count_opt, 8, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); return -4; }
    err = cudaMemcpyFromSymbol(out + 180, sol_fullencode_count_opt, 8, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (full_count): %s\n", cudaGetErrorString(err)); return -4; }

    // Copy done flag
    err = cudaMemcpyFromSymbol(out + 188, sol_done_opt, 4, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (done): %s\n", cudaGetErrorString(err)); return -4; }

    return 0;
}

// Cleanup function
extern "C" void sol_vanity_cleanup(int id) {
    if (id >= 0 && id < 16 && sol_gpu_contexts[id].initialized) {
        cudaSetDevice(id);
        cudaFree(sol_gpu_contexts[id].d_buffer);
        sol_gpu_contexts[id].d_buffer = nullptr;
        sol_gpu_contexts[id].initialized = false;
    }
}
