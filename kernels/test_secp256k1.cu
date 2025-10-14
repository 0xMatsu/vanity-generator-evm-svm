#include "secp256k1.h"
#include <stdio.h>

// Test kernel: generate ONE public key and copy to output
__global__ void test_one_key(uint8_t *private_key, uint8_t *public_key_out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GPU: Starting secp256k1 test...\n");

        uint8_t pubkey[65];
        secp256k1_get_public_key(private_key, pubkey);

        // Copy to output
        for (int i = 0; i < 65; i++) {
            public_key_out[i] = pubkey[i];
        }

        printf("GPU: Finished! PubKey[0-4]: %02x %02x %02x %02x %02x\n",
               pubkey[0], pubkey[1], pubkey[2], pubkey[3], pubkey[4]);
    }
}

extern "C" {
    void test_secp256k1_single_key(uint8_t *h_private_key, uint8_t *h_public_key) {
        uint8_t *d_private, *d_public;

        cudaMalloc(&d_private, 32);
        cudaMalloc(&d_public, 65);

        cudaMemcpy(d_private, h_private_key, 32, cudaMemcpyHostToDevice);

        test_one_key<<<1, 1>>>(d_private, d_public);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(h_public_key, d_public, 65, cudaMemcpyDeviceToHost);

        cudaFree(d_private);
        cudaFree(d_public);
    }
}
