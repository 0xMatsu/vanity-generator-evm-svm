// nvcc -o target/test-primitives kernels/test_primitives.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "base58_simple.cu"
#include "chacha_rng.h"

__global__ void test_primitives(const uint8_t *inputs, uint8_t *encoded, uint8_t *checks, uint32_t *block) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= 1024) return;
    const uint8_t *key = inputs + 32 * id;
    uint8_t *address = encoded + 45 * id;
    int len = (int)simple_base58_encode_32(key, address);
    bool ok = true;
    for (int n = 1; n <= len; ++n) {
        ok &= base58_suffix_match_mod_32(key, (const char*)address + len - n, n, false);
        char lower[45];
        for (int i = 0; i < n; ++i) {
            char c = address[len-n+i];
            lower[i] = c >= 'A' && c <= 'Z' ? c + 32 : c;
        }
        ok &= base58_suffix_match_mod_32(key, lower, n, true);
    }
    checks[id] = ok;
    if (id == 0) {
        uint32_t state[16] = {0x61707865,0x3320646e,0x79622d32,0x6b206574,
            0x03020100,0x07060504,0x0b0a0908,0x0f0e0d0c,0x13121110,0x17161514,0x1b1a1918,0x1f1e1d1c,
            1,0x09000000,0x4a000000,0};
        chacha_block(state, block);
    }
}

static void reference(const uint8_t *input, char *out) {
    const char *alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    unsigned char digits[45] = {0};
    int count = 0, zeros = 0;
    while (zeros < 32 && input[zeros] == 0) ++zeros;
    for (int i = zeros; i < 32; ++i) {
        unsigned carry = input[i];
        for (int j = 0; j < count; ++j) {
            carry += digits[j] * 256;
            digits[j] = carry % 58;
            carry /= 58;
        }
        while (carry) { digits[count++] = carry % 58; carry /= 58; }
    }
    for (int i = 0; i < zeros; ++i) out[i] = '1';
    for (int i = 0; i < count; ++i) out[zeros+i] = alphabet[digits[count-i-1]];
    out[zeros+count] = 0;
}

#define CUDA_CHECK(call) do { cudaError_t e = call; if(e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); return 1; } } while(0)
int main() {
    uint8_t inputs[1024*32], encoded[1024*45], checks[1024];
    uint32_t state = 42;
    for (int i = 0; i < 1024*32; ++i) { state = state * 1664525 + 1013904223; inputs[i] = state >> 24; }
    for (int i = 0; i <= 32; ++i) memset(inputs + 32*i, 0, i);
    memset(inputs+33*32, 255, 32);
    uint8_t *d_inputs, *d_encoded, *d_checks; uint32_t *d_block, block[16];
    CUDA_CHECK(cudaMalloc(&d_inputs, sizeof inputs));
    CUDA_CHECK(cudaMalloc(&d_encoded, sizeof encoded));
    CUDA_CHECK(cudaMalloc(&d_checks, sizeof checks));
    CUDA_CHECK(cudaMalloc(&d_block, sizeof block));
    CUDA_CHECK(cudaMemcpy(d_inputs, inputs, sizeof inputs, cudaMemcpyHostToDevice));
    test_primitives<<<32,32>>>(d_inputs,d_encoded,d_checks,d_block);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(encoded,d_encoded,sizeof encoded,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(checks,d_checks,sizeof checks,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(block,d_block,sizeof block,cudaMemcpyDeviceToHost));
    const uint32_t expected[16] = {0xe4e7f110,0x15593bd1,0x1fdd0f50,0xc47120a3,0xc7f4d1c7,0x0368c033,0x9aaa2204,0x4e6cd4c3,
        0x466482d2,0x09aa9f07,0x05d7c214,0xa2028bd9,0xd19c12b5,0xb94e16de,0xe883d0cb,0x4e3c50a2};
    if (memcmp(block,expected,sizeof block)) { fprintf(stderr,"ChaCha20 RFC vector failed\n"); return 1; }
    for (int i = 0; i < 1024; ++i) {
        char expected_address[45]; reference(inputs+32*i, expected_address);
        if (!checks[i] || strcmp(expected_address,(char*)encoded+45*i)) {
            fprintf(stderr,"Base58 differential test failed at %d\n",i); return 1;
        }
    }
    cudaFree(d_inputs); cudaFree(d_encoded); cudaFree(d_checks); cudaFree(d_block);
    puts("PASS: 1024 GPU Base58 vectors, all suffix lengths/case folding, RFC 8439 ChaCha20 vector");
}
