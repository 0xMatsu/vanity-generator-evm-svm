#include "base58.h"
#include <string.h>

// Simple base58 alphabet
__device__ const char BASE58_ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Divide eight big-endian 32-bit limbs by 58^5, emitting five digits per
// pass. At most nine passes replace the byte-at-a-time quadratic conversion.
__device__ ulong simple_base58_encode_32(const uint8_t *input, uint8_t *output) {
    uint32_t limbs[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        limbs[i] = ((uint32_t)input[4*i] << 24) | ((uint32_t)input[4*i+1] << 16)
            | ((uint32_t)input[4*i+2] << 8) | input[4*i+3];
    }
    int zeros = 0;
    while (zeros < 32 && input[zeros] == 0) ++zeros;
    uint8_t reversed[45];
    int digits = 0;
    int first = 0;
    while (first < 8 && limbs[first] == 0) ++first;
    while (first < 8) {
        uint64_t remainder = 0;
        for (int i = first; i < 8; ++i) {
            uint64_t value = (remainder << 32) | limbs[i];
            limbs[i] = (uint32_t)(value / 656356768ULL);
            remainder = value % 656356768ULL;
        }
        while (first < 8 && limbs[first] == 0) ++first;
        for (int i = 0; i < 5; ++i) {
            reversed[digits++] = BASE58_ALPHABET[remainder % 58];
            remainder /= 58;
            if (first == 8 && remainder == 0) break;
        }
    }
    for (int i = 0; i < zeros; ++i) output[i] = '1';
    for (int i = 0; i < digits; ++i) output[zeros+i] = reversed[digits-1-i];
    output[zeros+digits] = 0;
    return zeros+digits;
}

// Compute only the last `k` base58 characters (suffix) for a 32-byte input.
// Returns the number of characters written to `suffix_out` (min(k, total_len)).
// This avoids constructing the entire encoded string when only a suffix match is needed.
__device__ int simple_base58_suffix_32(const uint8_t *input, char *suffix_out, int k) {
    // Working buffer for the conversion
    uint8_t digits[64] = {0};
    int digit_count = 1;

    // Convert each input byte (same as full encode)
    for (int i = 0; i < 32; i++) {
        int carry = input[i];
        for (int j = 0; j < digit_count; j++) {
            carry += (int)digits[j] * 256;
            digits[j] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) {
            digits[digit_count++] = carry % 58;
            carry /= 58;
        }
    }

    // Note: leading zero bytes would add '1' characters at the front of the full string,
    // but they do not affect the last characters (suffix), so we can ignore them here.

    int out = 0;
    int limit = (k < digit_count) ? k : digit_count;
    for (int i = 0; i < limit; i++) {
        suffix_out[i] = BASE58_ALPHABET[digits[i]]; // digits are little-endian; index 0 is last char
        out++;
    }
    return out;
}

// Necessary suffix condition only: the caller must check the full encoding.
__device__ bool base58_suffix_match_mod_32(const uint8_t *input, const char *suffix, int suffix_len, bool case_insensitive) {
    const uint64_t mod = 58ULL * 58 * 58 * 58;
    uint64_t r = 0;
    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        uint32_t word = ((uint32_t)input[i] << 24) | ((uint32_t)input[i+1] << 16)
            | ((uint32_t)input[i+2] << 8) | input[i+3];
        r = ((r << 32) | word) % mod;
    }
    int k = suffix_len < 4 ? suffix_len : 4;
    for (int i = 0; i < k; ++i) {
        char actual = BASE58_ALPHABET[r % 58];
        char expected = suffix[suffix_len - 1 - i];
        if (case_insensitive) {
            if (actual >= 'A' && actual <= 'Z') actual += 32;
            if (expected >= 'A' && expected <= 'Z') expected += 32;
        }
        if (actual != expected) return false;
        r /= 58;
    }
    return true;
}
