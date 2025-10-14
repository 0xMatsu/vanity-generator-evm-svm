#ifndef KECCAK256_H
#define KECCAK256_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Keccak256 hash function
__device__ void keccak256(const uint8_t *input, size_t len, uint8_t *output);

#ifdef __cplusplus
}
#endif

#endif // KECCAK256_H
