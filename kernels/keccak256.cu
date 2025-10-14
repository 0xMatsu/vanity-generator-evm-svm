#include "keccak256.h"
#include <string.h>

// Keccak-256 implementation for CUDA
// Based on the FIPS 202 specification

#define KECCAK_ROUNDS 24

__device__ static const uint64_t keccak_round_constants[KECCAK_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccak_f1600(uint64_t state[25]) {
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        uint64_t C[5], D[5];

        // Theta
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[x + 5 * y] ^= D[x];
            }
        }

        // Rho and Pi
        uint64_t B[25];
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int r = ((x + 1) + (y + 1) * 2) % 5;
                int t = (x + 3 * y) % 5;

                // Rotation offsets for rho
                static const int rotation_offsets[5][5] = {
                    {0, 36, 3, 41, 18},
                    {1, 44, 10, 45, 2},
                    {62, 6, 43, 15, 61},
                    {28, 55, 25, 21, 56},
                    {27, 20, 39, 8, 14}
                };

                B[r + 5 * t] = rotl64(state[x + 5 * y], rotation_offsets[y][x]);
            }
        }

        // Chi
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                state[x + 5 * y] = B[x + 5 * y] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
            }
        }

        // Iota
        state[0] ^= keccak_round_constants[round];
    }
}

__device__ void keccak256(const uint8_t *input, size_t len, uint8_t *output) {
    uint64_t state[25] = {0};
    const size_t rate = 136; // 1088 bits = 136 bytes for SHA3-256/Keccak-256

    // Absorb phase
    size_t offset = 0;
    while (offset < len) {
        size_t block_size = (len - offset < rate) ? (len - offset) : rate;

        for (size_t i = 0; i < block_size; i++) {
            size_t state_index = i / 8;
            size_t byte_offset = i % 8;
            state[state_index] ^= ((uint64_t)input[offset + i]) << (8 * byte_offset);
        }

        if (block_size == rate) {
            keccak_f1600(state);
        }

        offset += block_size;
    }

    // Padding (Keccak uses 0x01 for domain separation, not SHA3's 0x06)
    size_t last_byte_index = len % rate;
    size_t state_index = last_byte_index / 8;
    size_t byte_offset = last_byte_index % 8;

    state[state_index] ^= ((uint64_t)0x01) << (8 * byte_offset);
    state[(rate - 1) / 8] ^= ((uint64_t)0x80) << (8 * ((rate - 1) % 8));

    keccak_f1600(state);

    // Squeeze phase - extract 256 bits (32 bytes)
    for (int i = 0; i < 32; i++) {
        output[i] = (state[i / 8] >> (8 * (i % 8))) & 0xFF;
    }
}
