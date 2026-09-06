// Standalone differential test, including the equal-x fallback at small scalars.
#include <cuda_runtime.h>
#include <stdio.h>
#include "secp256k1_mrspike.cu"

__global__ void verify_batch(int *failed) {
    uint8_t secret[32] = {0}, public_key[65], batch[SECP256K1_BATCH_SIZE*65];
    secret[31] = threadIdx.x + 1;
    if (threadIdx.x == 31) memset(secret, 42, 32);
    secp256k1_get_public_key(secret, public_key);
    secp256k1_public_add_batch(public_key, batch);
    for (int i = 0; i < SECP256K1_BATCH_SIZE; ++i) {
        for (int j = 31; j >= 0; --j) if (++secret[j]) break;
        secp256k1_get_public_key(secret, public_key);
        for (int j = 0; j < 65; ++j) {
            if (public_key[j] != batch[65*i+j]) atomicExch(failed, 1);
        }
    }
}
int main() {
    int *failed, result = 1;
    if (cudaMalloc(&failed, sizeof(int)) != cudaSuccess) return 1;
    if (cudaMemset(failed, 0, sizeof(int)) != cudaSuccess) return 1;
    verify_batch<<<1,32>>>(failed);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    if (cudaMemcpy(&result, failed, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    cudaFree(failed);
    if (result) { puts("FAIL: batched points differ from scalar multiplication"); return 1; }
    printf("PASS: %d batched points match independent scalar multiplication\n", 32*SECP256K1_BATCH_SIZE);
}
