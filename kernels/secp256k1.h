#ifndef SECP256K1_H
#define SECP256K1_H

#include <stdint.h>
#ifndef SECP256K1_BATCH_SIZE
#define SECP256K1_BATCH_SIZE 32
#endif
static_assert(SECP256K1_BATCH_SIZE == 8 || SECP256K1_BATCH_SIZE == 16 || SECP256K1_BATCH_SIZE == 32,
    "batch size must be 8, 16 or 32");

#ifdef __cplusplus
extern "C" {
#endif

// secp256k1 curve parameters
// y^2 = x^3 + 7 over F_p where p = 2^256 - 2^32 - 977

// Generate secp256k1 public key from private key
__device__ void secp256k1_get_public_key(const uint8_t *private_key, uint8_t *public_key);

// Output holds BATCH_SIZE public keys; input scalar must be in [1, n-BATCH_SIZE-1].
__device__ void secp256k1_public_add_batch(const uint8_t *in, uint8_t *out);

// Add generator point G to an existing uncompressed public key (0x04 || X || Y)
// out_public_key may alias in_public_key for in-place update
__device__ void secp256k1_public_add_generator(const uint8_t *in_public_key, uint8_t *out_public_key);

#ifdef __cplusplus
}
#endif

#endif // SECP256K1_H
