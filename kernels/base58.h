#ifndef BS58_H
#define BS58_H

#include <stdint.h>

// Windows MSVC doesn't define ulong and uint, so define them
#ifndef __linux__
typedef unsigned long ulong;
typedef unsigned int uint;
#endif

__device__ ulong fd_base58_encode_32(uint8_t *bytes, uint8_t *out, bool case_insensitive);

// Simple, correct base58 encoder
__device__ ulong simple_base58_encode_32(const uint8_t *input, uint8_t *output);

// Suffix-only base58 helper for 32-byte input
__device__ int simple_base58_suffix_32(const uint8_t *input, char *suffix_out, int k);

// Fast suffix check via modular reduction: computes N mod 58^k and compares last k digits
__device__ bool base58_suffix_match_mod_32(const uint8_t *input, const char *suffix, int suffix_len, bool case_insensitive);

#endif
