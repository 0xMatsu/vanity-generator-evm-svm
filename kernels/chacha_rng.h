#pragma once
#include <stdint.h>

// ChaCha20 block function (RFC 8439), with a 64-bit counter and 64-bit
// stream id. Each launch receives a fresh 256-bit key from the host CSPRNG.
#define CHACHA_HD __host__ __device__
CHACHA_HD inline uint32_t chacha_rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}
CHACHA_HD inline void chacha_qr(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
    a += b; d = chacha_rotl(d ^ a, 16);
    c += d; b = chacha_rotl(b ^ c, 12);
    a += b; d = chacha_rotl(d ^ a, 8);
    c += d; b = chacha_rotl(b ^ c, 7);
}
CHACHA_HD inline void chacha_block(const uint32_t state[16], uint32_t out[16]) {
    for (int i = 0; i < 16; ++i) out[i] = state[i];
    for (int i = 0; i < 10; ++i) {
        chacha_qr(out[0], out[4], out[8], out[12]);
        chacha_qr(out[1], out[5], out[9], out[13]);
        chacha_qr(out[2], out[6], out[10], out[14]);
        chacha_qr(out[3], out[7], out[11], out[15]);
        chacha_qr(out[0], out[5], out[10], out[15]);
        chacha_qr(out[1], out[6], out[11], out[12]);
        chacha_qr(out[2], out[7], out[8], out[13]);
        chacha_qr(out[3], out[4], out[9], out[14]);
    }
    for (int i = 0; i < 16; ++i) out[i] += state[i];
}
struct ChaChaRng {
    uint32_t state[16], words[16];
    int offset;
    CHACHA_HD ChaChaRng(const uint8_t key[32], uint64_t stream) : offset(16) {
        state[0] = 0x61707865; state[1] = 0x3320646e;
        state[2] = 0x79622d32; state[3] = 0x6b206574;
        for (int i = 0; i < 8; ++i) {
            state[4+i] = (uint32_t)key[4*i] | ((uint32_t)key[4*i+1] << 8)
                | ((uint32_t)key[4*i+2] << 16) | ((uint32_t)key[4*i+3] << 24);
        }
        state[12] = state[13] = 0;
        state[14] = (uint32_t)stream; state[15] = (uint32_t)(stream >> 32);
    }
    CHACHA_HD void next32(uint8_t out[32]) {
        if (offset == 16) {
            chacha_block(state, words);
            if (++state[12] == 0) ++state[13];
            offset = 0;
        }
        for (int i = 0; i < 8; ++i) {
            uint32_t w = words[offset + i];
            for (int j = 0; j < 4; ++j) out[4*i+j] = (uint8_t)(w >> (8*j));
        }
        offset += 8;
    }
};
#undef CHACHA_HD
