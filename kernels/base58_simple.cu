#include "base58.h"
#include <string.h>

// Simple base58 alphabet
__device__ const char BASE58_ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Simple base58 encode for 32 bytes
// This is a straightforward implementation that matches how bs58 crate works
__device__ ulong simple_base58_encode_32(const uint8_t *input, uint8_t *output) {
    // Working buffer for the conversion - use large integer arithmetic
    uint8_t digits[64] = {0};  // Enough for base58 of 32 bytes
    int digit_count = 1;

    // Convert each input byte
    for (int i = 0; i < 32; i++) {
        int carry = input[i];

        // Add carry to existing digits (multiply by 256 and add)
        for (int j = 0; j < digit_count; j++) {
            carry += (int)digits[j] * 256;
            digits[j] = carry % 58;
            carry /= 58;
        }

        // Add new digits for remaining carry
        while (carry > 0) {
            digits[digit_count++] = carry % 58;
            carry /= 58;
        }
    }

    // Count leading zeros in input
    int leading_zeros = 0;
    for (int i = 0; i < 32; i++) {
        if (input[i] == 0) {
            leading_zeros++;
        } else {
            break;
        }
    }

    // Build output string
    int out_len = 0;

    // Add '1' for each leading zero byte
    for (int i = 0; i < leading_zeros; i++) {
        output[out_len++] = '1';
    }

    // Add base58 digits in reverse order (big-endian)
    for (int i = digit_count - 1; i >= 0; i--) {
        output[out_len++] = BASE58_ALPHABET[digits[i]];
    }

    output[out_len] = '\0';
    return out_len;
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

__device__ static inline int b58_index(char c) {
    // Linear scan over alphabet; k is small so cost is minor
    for (int i = 0; i < 58; ++i) {
        if (BASE58_ALPHABET[i] == c) return i;
    }
    return -1;
}

__device__ bool base58_suffix_match_mod_32(const uint8_t *input, const char *suffix, int suffix_len, bool case_insensitive) {
    // Limit modulus to 58^k fitting in 64-bit; k up to 10 is safe
    int k = suffix_len;
    if (k <= 0) return true;
    if (k > 10) return false; // fall back to full encode in caller

    uint64_t mod = 1;
    for (int i = 0; i < k; ++i) mod *= 58ULL;

    // Compute r = N mod 58^k, N as big-endian integer from 32-byte input
    uint64_t r = 0;
    for (int i = 0; i < 32; ++i) {
        r = (r * 256ULL + (uint64_t)input[i]) % mod;
    }

    // Compare last k digits from least significant (end of string)
    for (int i = 0; i < k; ++i) {
        int d = (int)(r % 58ULL);
        r /= 58ULL;
        char sc = suffix[suffix_len - 1 - i];
        if (case_insensitive && sc >= 'A' && sc <= 'Z') sc = sc + 32;
        // Find index of sc (case-sensitive alphabet)
        int sc_idx = b58_index(sc);
        if (sc_idx < 0) return false;
        if (d != sc_idx) return false;
    }
    return true;
}
