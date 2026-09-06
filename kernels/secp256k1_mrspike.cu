/*
 * secp256k1 implementation based on MrSpike63's proven working vanity-eth-address
 * Uses affine coordinates (simpler than Jacobian)
 * Structure adapted from their GPU implementation with PTX assembly math
 */

#include "secp256k1.h"
#include <stdint.h>
#include <string.h>

// Use MrSpike63's structure-based approach (with underscore prefix to match eth_math_working.h)
struct _uint256 {
    uint32_t a, b, c, d, e, f, g, h;
};

struct _uint256c {
    bool carry;
    uint32_t a, b, c, d, e, f, g, h;
};

struct CurvePoint {
    _uint256 x;
    _uint256 y;
};

// secp256k1 constants
__device__ __constant__ _uint256 SECP_P = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

__device__ __constant__ _uint256 SECP_N = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE,
    0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

// Generator point G
__device__ __constant__ _uint256 G_X = {
    0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
    0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};

__device__ __constant__ _uint256 G_Y = {
    0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
    0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};

// Helper: convert uint32 to _uint256
__device__ inline _uint256 uint32_to_uint256(uint32_t x) {
    return _uint256{0, 0, 0, 0, 0, 0, 0, x};
}

// Helper: convert _uint256 to _uint256c
__device__ inline _uint256c uint256_to_uint256c(_uint256 x) {
    return _uint256c{0, x.a, x.b, x.c, x.d, x.e, x.f, x.g, x.h};
}

// Comparison: x >= y
__device__ inline bool gte_256(_uint256 x, _uint256 y) {
    bool gte = false;
    bool equal = true;
    gte |= (x.a > y.a); equal &= (x.a == y.a);
    gte |= ((x.b > y.b) && equal); equal &= (x.b == y.b);
    gte |= ((x.c > y.c) && equal); equal &= (x.c == y.c);
    gte |= ((x.d > y.d) && equal); equal &= (x.d == y.d);
    gte |= ((x.e > y.e) && equal); equal &= (x.e == y.e);
    gte |= ((x.f > y.f) && equal); equal &= (x.f == y.f);
    gte |= ((x.g > y.g) && equal); equal &= (x.g == y.g);
    gte |= ((x.h >= y.h) && equal);
    return gte;
}

// Comparison: x == y
__device__ inline bool eqeq_256(_uint256 x, _uint256 y) {
    return x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d &&
           x.e == y.e && x.f == y.f && x.g == y.g && x.h == y.h;
}

// Import PTX assembly math functions from eth_math_working.h
// Note: This defines add_256_with_c, sub_256_mod_p, mul_256_mod_p, eeuclid_256_mod_p
#include "eth_math_working.h"

// Add with modular reduction (eth_math doesn't have this)
__device__ _uint256 add_256_mod_p(_uint256 x, _uint256 y) {
    _uint256c result_c = add_256_with_c(x, y);
    _uint256 result;
    result.a = result_c.a;
    result.b = result_c.b;
    result.c = result_c.c;
    result.d = result_c.d;
    result.e = result_c.e;
    result.f = result_c.f;
    result.g = result_c.g;
    result.h = result_c.h;

    // If carry or result >= P, subtract P
    if (result_c.carry || gte_256(result, SECP_P)) {
        return sub_256_mod_p(result, SECP_P);
    }
    return result;
}

// Point doubling lambda: (3 * x^2) / (2 * y)
__device__ _uint256 point_double_lambda(CurvePoint p) {
    // 3 * x^2
    _uint256 x_sq = mul_256_mod_p(p.x, p.x);
    _uint256 three = uint32_to_uint256(3);
    _uint256 three_x_sq = mul_256_mod_p(three, x_sq);

    // 2 * y
    _uint256 two = uint32_to_uint256(2);
    _uint256 two_y = mul_256_mod_p(two, p.y);

    // (3 * x^2) / (2 * y)
    return mul_256_mod_p(three_x_sq, eeuclid_256_mod_p(two_y));
}

// Point addition lambda: (q.y - p.y) / (q.x - p.x)
__device__ _uint256 point_add_lambda(CurvePoint p, CurvePoint q) {
    _uint256 dy = sub_256_mod_p(q.y, p.y);
    _uint256 dx = sub_256_mod_p(q.x, p.x);
    return mul_256_mod_p(dy, eeuclid_256_mod_p(dx));
}

// Unified point addition (handles both add and double)
__device__ CurvePoint point_add(CurvePoint p, CurvePoint q) {
    _uint256 lambda;

    // If same x coordinate, it's point doubling
    if (eqeq_256(p.x, q.x)) {
        lambda = point_double_lambda(p);
    } else {
        lambda = point_add_lambda(p, q);
    }

    // x_r = lambda^2 - p.x - q.x
    _uint256 lambda_sq = mul_256_mod_p(lambda, lambda);
    _uint256 x_r = sub_256_mod_p(sub_256_mod_p(lambda_sq, p.x), q.x);

    // y_r = lambda * (p.x - x_r) - p.y
    _uint256 dx = sub_256_mod_p(p.x, x_r);
    _uint256 y_r = sub_256_mod_p(mul_256_mod_p(lambda, dx), p.y);

    return CurvePoint{x_r, y_r};
}

// Scalar multiplication: result = scalar * point
// Uses double-and-add from LSB to MSB (like MrSpike63)
__device__ CurvePoint point_multiply(CurvePoint point, _uint256 scalar) {
    CurvePoint result;
    bool at_infinity = true;
    CurvePoint temp = point;

    // Process h (bits 0-31)
    for (int i = 0; i < 32; i++) {
        if (scalar.h & (1U << i)) {
            if (at_infinity) {
                result = temp;
                at_infinity = false;
            } else {
                result = point_add(result, temp);
            }
        }
        temp = point_add(temp, temp);  // Double
    }

    // Process g (bits 32-63)
    for (int i = 0; i < 32; i++) {
        if (scalar.g & (1U << i)) {
            if (at_infinity) {
                result = temp;
                at_infinity = false;
            } else {
                result = point_add(result, temp);
            }
        }
        temp = point_add(temp, temp);
    }

    // Process f (bits 64-95)
    for (int i = 0; i < 32; i++) {
        if (scalar.f & (1U << i)) {
            if (at_infinity) {
                result = temp;
                at_infinity = false;
            } else {
                result = point_add(result, temp);
            }
        }
        temp = point_add(temp, temp);
    }

    // Process e (bits 96-127)
    for (int i = 0; i < 32; i++) {
        if (scalar.e & (1U << i)) {
            if (at_infinity) {
                result = temp;
                at_infinity = false;
            } else {
                result = point_add(result, temp);
            }
        }
        temp = point_add(temp, temp);
    }

    // Process d (bits 128-159)
    for (int i = 0; i < 32; i++) {
        if (scalar.d & (1U << i)) {
            if (at_infinity) {
                result = temp;
                at_infinity = false;
            } else {
                result = point_add(result, temp);
            }
        }
        temp = point_add(temp, temp);
    }

    // Process c (bits 160-191)
    for (int i = 0; i < 32; i++) {
        if (scalar.c & (1U << i)) {
            if (at_infinity) {
                result = temp;
                at_infinity = false;
            } else {
                result = point_add(result, temp);
            }
        }
        temp = point_add(temp, temp);
    }

    // Process b (bits 192-223)
    for (int i = 0; i < 32; i++) {
        if (scalar.b & (1U << i)) {
            if (at_infinity) {
                result = temp;
                at_infinity = false;
            } else {
                result = point_add(result, temp);
            }
        }
        temp = point_add(temp, temp);
    }

    // Process a (bits 224-255)
    for (int i = 0; i < 32; i++) {
        if (scalar.a & (1U << i)) {
            if (at_infinity) {
                result = temp;
                at_infinity = false;
            } else {
                result = point_add(result, temp);
            }
        }
        temp = point_add(temp, temp);
    }

    return result;
}

// Main entry point: generate public key from private key
__device__ void secp256k1_get_public_key(const uint8_t *private_key, uint8_t *public_key) {
    // Convert private key bytes to _uint256 (big-endian)
    _uint256 scalar;
    scalar.a = (private_key[0] << 24) | (private_key[1] << 16) | (private_key[2] << 8) | private_key[3];
    scalar.b = (private_key[4] << 24) | (private_key[5] << 16) | (private_key[6] << 8) | private_key[7];
    scalar.c = (private_key[8] << 24) | (private_key[9] << 16) | (private_key[10] << 8) | private_key[11];
    scalar.d = (private_key[12] << 24) | (private_key[13] << 16) | (private_key[14] << 8) | private_key[15];
    scalar.e = (private_key[16] << 24) | (private_key[17] << 16) | (private_key[18] << 8) | private_key[19];
    scalar.f = (private_key[20] << 24) | (private_key[21] << 16) | (private_key[22] << 8) | private_key[23];
    scalar.g = (private_key[24] << 24) | (private_key[25] << 16) | (private_key[26] << 8) | private_key[27];
    scalar.h = (private_key[28] << 24) | (private_key[29] << 16) | (private_key[30] << 8) | private_key[31];

    // Create generator point
    CurvePoint G = {G_X, G_Y};

    // Compute public_key = scalar * G
    CurvePoint pubkey = point_multiply(G, scalar);

    // Convert to uncompressed format: 0x04 || x || y (big-endian)
    public_key[0] = 0x04;

    // X coordinate (32 bytes)
    public_key[1] = (pubkey.x.a >> 24) & 0xFF;
    public_key[2] = (pubkey.x.a >> 16) & 0xFF;
    public_key[3] = (pubkey.x.a >> 8) & 0xFF;
    public_key[4] = pubkey.x.a & 0xFF;

    public_key[5] = (pubkey.x.b >> 24) & 0xFF;
    public_key[6] = (pubkey.x.b >> 16) & 0xFF;
    public_key[7] = (pubkey.x.b >> 8) & 0xFF;
    public_key[8] = pubkey.x.b & 0xFF;

    public_key[9] = (pubkey.x.c >> 24) & 0xFF;
    public_key[10] = (pubkey.x.c >> 16) & 0xFF;
    public_key[11] = (pubkey.x.c >> 8) & 0xFF;
    public_key[12] = pubkey.x.c & 0xFF;

    public_key[13] = (pubkey.x.d >> 24) & 0xFF;
    public_key[14] = (pubkey.x.d >> 16) & 0xFF;
    public_key[15] = (pubkey.x.d >> 8) & 0xFF;
    public_key[16] = pubkey.x.d & 0xFF;

    public_key[17] = (pubkey.x.e >> 24) & 0xFF;
    public_key[18] = (pubkey.x.e >> 16) & 0xFF;
    public_key[19] = (pubkey.x.e >> 8) & 0xFF;
    public_key[20] = pubkey.x.e & 0xFF;

    public_key[21] = (pubkey.x.f >> 24) & 0xFF;
    public_key[22] = (pubkey.x.f >> 16) & 0xFF;
    public_key[23] = (pubkey.x.f >> 8) & 0xFF;
    public_key[24] = pubkey.x.f & 0xFF;

    public_key[25] = (pubkey.x.g >> 24) & 0xFF;
    public_key[26] = (pubkey.x.g >> 16) & 0xFF;
    public_key[27] = (pubkey.x.g >> 8) & 0xFF;
    public_key[28] = pubkey.x.g & 0xFF;

    public_key[29] = (pubkey.x.h >> 24) & 0xFF;
    public_key[30] = (pubkey.x.h >> 16) & 0xFF;
    public_key[31] = (pubkey.x.h >> 8) & 0xFF;
    public_key[32] = pubkey.x.h & 0xFF;

    // Y coordinate (32 bytes)
    public_key[33] = (pubkey.y.a >> 24) & 0xFF;
    public_key[34] = (pubkey.y.a >> 16) & 0xFF;
    public_key[35] = (pubkey.y.a >> 8) & 0xFF;
    public_key[36] = pubkey.y.a & 0xFF;

    public_key[37] = (pubkey.y.b >> 24) & 0xFF;
    public_key[38] = (pubkey.y.b >> 16) & 0xFF;
    public_key[39] = (pubkey.y.b >> 8) & 0xFF;
    public_key[40] = pubkey.y.b & 0xFF;

    public_key[41] = (pubkey.y.c >> 24) & 0xFF;
    public_key[42] = (pubkey.y.c >> 16) & 0xFF;
    public_key[43] = (pubkey.y.c >> 8) & 0xFF;
    public_key[44] = pubkey.y.c & 0xFF;

    public_key[45] = (pubkey.y.d >> 24) & 0xFF;
    public_key[46] = (pubkey.y.d >> 16) & 0xFF;
    public_key[47] = (pubkey.y.d >> 8) & 0xFF;
    public_key[48] = pubkey.y.d & 0xFF;

    public_key[49] = (pubkey.y.e >> 24) & 0xFF;
    public_key[50] = (pubkey.y.e >> 16) & 0xFF;
    public_key[51] = (pubkey.y.e >> 8) & 0xFF;
    public_key[52] = pubkey.y.e & 0xFF;

    public_key[53] = (pubkey.y.f >> 24) & 0xFF;
    public_key[54] = (pubkey.y.f >> 16) & 0xFF;
    public_key[55] = (pubkey.y.f >> 8) & 0xFF;
    public_key[56] = pubkey.y.f & 0xFF;

    public_key[57] = (pubkey.y.g >> 24) & 0xFF;
    public_key[58] = (pubkey.y.g >> 16) & 0xFF;
    public_key[59] = (pubkey.y.g >> 8) & 0xFF;
    public_key[60] = pubkey.y.g & 0xFF;

    public_key[61] = (pubkey.y.h >> 24) & 0xFF;
    public_key[62] = (pubkey.y.h >> 16) & 0xFF;
    public_key[63] = (pubkey.y.h >> 8) & 0xFF;
    public_key[64] = pubkey.y.h & 0xFF;
}

// Add generator point G to an existing uncompressed public key (0x04 || X || Y)
__device__ void secp256k1_public_add_generator(const uint8_t *in_public_key, uint8_t *out_public_key) {
    // Parse input public key (uncompressed)
    CurvePoint pubPt;
    pubPt.x.a = (in_public_key[1] << 24) | (in_public_key[2] << 16) | (in_public_key[3] << 8) | in_public_key[4];
    pubPt.x.b = (in_public_key[5] << 24) | (in_public_key[6] << 16) | (in_public_key[7] << 8) | in_public_key[8];
    pubPt.x.c = (in_public_key[9] << 24) | (in_public_key[10] << 16) | (in_public_key[11] << 8) | in_public_key[12];
    pubPt.x.d = (in_public_key[13] << 24) | (in_public_key[14] << 16) | (in_public_key[15] << 8) | in_public_key[16];
    pubPt.x.e = (in_public_key[17] << 24) | (in_public_key[18] << 16) | (in_public_key[19] << 8) | in_public_key[20];
    pubPt.x.f = (in_public_key[21] << 24) | (in_public_key[22] << 16) | (in_public_key[23] << 8) | in_public_key[24];
    pubPt.x.g = (in_public_key[25] << 24) | (in_public_key[26] << 16) | (in_public_key[27] << 8) | in_public_key[28];
    pubPt.x.h = (in_public_key[29] << 24) | (in_public_key[30] << 16) | (in_public_key[31] << 8) | in_public_key[32];

    pubPt.y.a = (in_public_key[33] << 24) | (in_public_key[34] << 16) | (in_public_key[35] << 8) | in_public_key[36];
    pubPt.y.b = (in_public_key[37] << 24) | (in_public_key[38] << 16) | (in_public_key[39] << 8) | in_public_key[40];
    pubPt.y.c = (in_public_key[41] << 24) | (in_public_key[42] << 16) | (in_public_key[43] << 8) | in_public_key[44];
    pubPt.y.d = (in_public_key[45] << 24) | (in_public_key[46] << 16) | (in_public_key[47] << 8) | in_public_key[48];
    pubPt.y.e = (in_public_key[49] << 24) | (in_public_key[50] << 16) | (in_public_key[51] << 8) | in_public_key[52];
    pubPt.y.f = (in_public_key[53] << 24) | (in_public_key[54] << 16) | (in_public_key[55] << 8) | in_public_key[56];
    pubPt.y.g = (in_public_key[57] << 24) | (in_public_key[58] << 16) | (in_public_key[59] << 8) | in_public_key[60];
    pubPt.y.h = (in_public_key[61] << 24) | (in_public_key[62] << 16) | (in_public_key[63] << 8) | in_public_key[64];

    // Q = pubPt + G
    CurvePoint genPt = {G_X, G_Y};
    CurvePoint Q = point_add(pubPt, genPt);

    // Write out uncompressed format
    out_public_key[0] = 0x04;
    // X
    out_public_key[1] = (Q.x.a >> 24) & 0xFF;
    out_public_key[2] = (Q.x.a >> 16) & 0xFF;
    out_public_key[3] = (Q.x.a >> 8) & 0xFF;
    out_public_key[4] = Q.x.a & 0xFF;
    out_public_key[5] = (Q.x.b >> 24) & 0xFF;
    out_public_key[6] = (Q.x.b >> 16) & 0xFF;
    out_public_key[7] = (Q.x.b >> 8) & 0xFF;
    out_public_key[8] = Q.x.b & 0xFF;
    out_public_key[9] = (Q.x.c >> 24) & 0xFF;
    out_public_key[10] = (Q.x.c >> 16) & 0xFF;
    out_public_key[11] = (Q.x.c >> 8) & 0xFF;
    out_public_key[12] = Q.x.c & 0xFF;
    out_public_key[13] = (Q.x.d >> 24) & 0xFF;
    out_public_key[14] = (Q.x.d >> 16) & 0xFF;
    out_public_key[15] = (Q.x.d >> 8) & 0xFF;
    out_public_key[16] = Q.x.d & 0xFF;
    out_public_key[17] = (Q.x.e >> 24) & 0xFF;
    out_public_key[18] = (Q.x.e >> 16) & 0xFF;
    out_public_key[19] = (Q.x.e >> 8) & 0xFF;
    out_public_key[20] = Q.x.e & 0xFF;
    out_public_key[21] = (Q.x.f >> 24) & 0xFF;
    out_public_key[22] = (Q.x.f >> 16) & 0xFF;
    out_public_key[23] = (Q.x.f >> 8) & 0xFF;
    out_public_key[24] = Q.x.f & 0xFF;
    out_public_key[25] = (Q.x.g >> 24) & 0xFF;
    out_public_key[26] = (Q.x.g >> 16) & 0xFF;
    out_public_key[27] = (Q.x.g >> 8) & 0xFF;
    out_public_key[28] = Q.x.g & 0xFF;
    out_public_key[29] = (Q.x.h >> 24) & 0xFF;
    out_public_key[30] = (Q.x.h >> 16) & 0xFF;
    out_public_key[31] = (Q.x.h >> 8) & 0xFF;
    out_public_key[32] = Q.x.h & 0xFF;
    // Y
    out_public_key[33] = (Q.y.a >> 24) & 0xFF;
    out_public_key[34] = (Q.y.a >> 16) & 0xFF;
    out_public_key[35] = (Q.y.a >> 8) & 0xFF;
    out_public_key[36] = Q.y.a & 0xFF;
    out_public_key[37] = (Q.y.b >> 24) & 0xFF;
    out_public_key[38] = (Q.y.b >> 16) & 0xFF;
    out_public_key[39] = (Q.y.b >> 8) & 0xFF;
    out_public_key[40] = Q.y.b & 0xFF;
    out_public_key[41] = (Q.y.c >> 24) & 0xFF;
    out_public_key[42] = (Q.y.c >> 16) & 0xFF;
    out_public_key[43] = (Q.y.c >> 8) & 0xFF;
    out_public_key[44] = Q.y.c & 0xFF;
    out_public_key[45] = (Q.y.d >> 24) & 0xFF;
    out_public_key[46] = (Q.y.d >> 16) & 0xFF;
    out_public_key[47] = (Q.y.d >> 8) & 0xFF;
    out_public_key[48] = Q.y.d & 0xFF;
    out_public_key[49] = (Q.y.e >> 24) & 0xFF;
    out_public_key[50] = (Q.y.e >> 16) & 0xFF;
    out_public_key[51] = (Q.y.e >> 8) & 0xFF;
    out_public_key[52] = Q.y.e & 0xFF;
    out_public_key[53] = (Q.y.f >> 24) & 0xFF;
    out_public_key[54] = (Q.y.f >> 16) & 0xFF;
    out_public_key[55] = (Q.y.f >> 8) & 0xFF;
    out_public_key[56] = Q.y.f & 0xFF;
    out_public_key[57] = (Q.y.g >> 24) & 0xFF;
    out_public_key[58] = (Q.y.g >> 16) & 0xFF;
    out_public_key[59] = (Q.y.g >> 8) & 0xFF;
    out_public_key[60] = Q.y.g & 0xFF;
    out_public_key[61] = (Q.y.h >> 24) & 0xFF;
    out_public_key[62] = (Q.y.h >> 16) & 0xFF;
    out_public_key[63] = (Q.y.h >> 8) & 0xFF;
    out_public_key[64] = Q.y.h & 0xFF;
}

// Multiples G..32G, computed by the secp256k1 group law.
__device__ __constant__ CurvePoint BATCH_G[32] = {
    {{0x79be667e, 0xf9dcbbac, 0x55a06295, 0xce870b07, 0x029bfcdb, 0x2dce28d9, 0x59f2815b, 0x16f81798},{0x483ada77, 0x26a3c465, 0x5da4fbfc, 0x0e1108a8, 0xfd17b448, 0xa6855419, 0x9c47d08f, 0xfb10d4b8}},
    {{0xc6047f94, 0x41ed7d6d, 0x3045406e, 0x95c07cd8, 0x5c778e4b, 0x8cef3ca7, 0xabac09b9, 0x5c709ee5},{0x1ae168fe, 0xa63dc339, 0xa3c58419, 0x466ceaee, 0xf7f63265, 0x3266d0e1, 0x236431a9, 0x50cfe52a}},
    {{0xf9308a01, 0x9258c310, 0x49344f85, 0xf89d5229, 0xb531c845, 0x836f99b0, 0x8601f113, 0xbce036f9},{0x388f7b0f, 0x632de814, 0x0fe337e6, 0x2a37f356, 0x6500a999, 0x34c2231b, 0x6cb9fd75, 0x84b8e672}},
    {{0xe493dbf1, 0xc10d80f3, 0x581e4904, 0x930b1404, 0xcc6c1390, 0x0ee07584, 0x74fa94ab, 0xe8c4cd13},{0x51ed993e, 0xa0d455b7, 0x5642e209, 0x8ea51448, 0xd967ae33, 0xbfbdfe40, 0xcfe97bdc, 0x47739922}},
    {{0x2f8bde4d, 0x1a072093, 0x55b4a725, 0x0a5c5128, 0xe88b84bd, 0xdc619ab7, 0xcba8d569, 0xb240efe4},{0xd8ac2226, 0x36e5e3d6, 0xd4dba9dd, 0xa6c9c426, 0xf788271b, 0xab0d6840, 0xdca87d3a, 0xa6ac62d6}},
    {{0xfff97bd5, 0x755eeea4, 0x20453a14, 0x355235d3, 0x82f6472f, 0x8568a18b, 0x2f057a14, 0x60297556},{0xae12777a, 0xacfbb620, 0xf3be9601, 0x7f45c560, 0xde80f0f6, 0x518fe4a0, 0x3c870c36, 0xb075f297}},
    {{0x5cbdf064, 0x6e5db4ea, 0xa398f365, 0xf2ea7a0e, 0x3d419b7e, 0x0330e39c, 0xe92bdded, 0xcac4f9bc},{0x6aebca40, 0xba255960, 0xa3178d6d, 0x861a54db, 0xa813d0b8, 0x13fde7b5, 0xa5082628, 0x087264da}},
    {{0x2f01e5e1, 0x5cca351d, 0xaff3843f, 0xb70f3c2f, 0x0a1bdd05, 0xe5af888a, 0x67784ef3, 0xe10a2a01},{0x5c4da8a7, 0x41539949, 0x293d082a, 0x132d13b4, 0xc2e213d6, 0xba5b7617, 0xb5da2cb7, 0x6cbde904}},
    {{0xacd484e2, 0xf0c7f653, 0x09ad178a, 0x9f559abd, 0xe0979697, 0x4c57e714, 0xc35f110d, 0xfc27ccbe},{0xcc338921, 0xb0a7d9fd, 0x64380971, 0x763b61e9, 0xadd888a4, 0x375f8e0f, 0x05cc262a, 0xc64f9c37}},
    {{0xa0434d9e, 0x47f3c862, 0x35477c7b, 0x1ae6ae5d, 0x3442d49b, 0x1943c2b7, 0x52a68e2a, 0x47e247c7},{0x893aba42, 0x5419bc27, 0xa3b6c7e6, 0x93a24c69, 0x6f794c2e, 0xd877a159, 0x3cbee53b, 0x037368d7}},
    {{0x774ae7f8, 0x58a9411e, 0x5ef4246b, 0x70c65aac, 0x5649980b, 0xe5c17891, 0xbbec1789, 0x5da008cb},{0xd984a032, 0xeb6b5e19, 0x0243dd56, 0xd7b7b365, 0x372db1e2, 0xdff9d6a8, 0x301d74c9, 0xc953c61b}},
    {{0xd01115d5, 0x48e7561b, 0x15c38f00, 0x4d734633, 0x687cf441, 0x9620095b, 0xc5b0f470, 0x70afe85a},{0xa9f34ffd, 0xc815e0d7, 0xa8b64537, 0xe17bd815, 0x79238c5d, 0xd9a86d52, 0x6b051b13, 0xf4062327}},
    {{0xf28773c2, 0xd975288b, 0xc7d1d205, 0xc3748651, 0xb075fbc6, 0x610e58cd, 0xdeeddf8f, 0x19405aa8},{0x0ab0902e, 0x8d880a89, 0x758212eb, 0x65cdaf47, 0x3a1a06da, 0x521fa91f, 0x29b5cb52, 0xdb03ed81}},
    {{0x499fdf9e, 0x895e719c, 0xfd64e67f, 0x07d38e32, 0x26aa7b63, 0x678949e6, 0xe49b241a, 0x60e823e4},{0xcac2f6c4, 0xb54e8551, 0x90f044e4, 0xa7b3d464, 0x464279c2, 0x7a3f95bc, 0xc65f40d4, 0x03a13f5b}},
    {{0xd7924d4f, 0x7d43ea96, 0x5a465ae3, 0x095ff411, 0x31e5946f, 0x3c85f79e, 0x44adbcf8, 0xe27e080e},{0x581e2872, 0xa86c72a6, 0x83842ec2, 0x28cc6def, 0xea40af2b, 0xd896d3a5, 0xc504dc9f, 0xf6a26b58}},
    {{0xe60fce93, 0xb59e9ec5, 0x3011aabc, 0x21c23e97, 0xb2a31369, 0xb87a5ae9, 0xc44ee89e, 0x2a6dec0a},{0xf7e35073, 0x99e59592, 0x9db99f34, 0xf5793710, 0x1296891e, 0x44d23f0b, 0xe1f32cce, 0x69616821}},
    {{0xdefdea4c, 0xdb677750, 0xa420fee8, 0x07eacf21, 0xeb9898ae, 0x79b97687, 0x66e4faa0, 0x4a2d4a34},{0x4211ab06, 0x94635168, 0xe997b0ea, 0xd2a93dae, 0xced1f4a0, 0x4a95c0f6, 0xcfb199f6, 0x9e56eb77}},
    {{0x5601570c, 0xb47f238d, 0x2b0286db, 0x4a990fa0, 0xf3ba28d1, 0xa319f5e7, 0xcf55c2a2, 0x444da7cc},{0xc136c1dc, 0x0cbeb930, 0xe9e29804, 0x3589351d, 0x81d8e0bc, 0x736ae2a1, 0xf5192e5e, 0x8b061d58}},
    {{0x2b4ea0a7, 0x97a443d2, 0x93ef5cff, 0x444f4979, 0xf06acfeb, 0xd7e86d27, 0x74756561, 0x38385b6c},{0x85e89bc0, 0x37945d93, 0xb343083b, 0x5a1c8613, 0x1a01f60c, 0x50269763, 0xb570c854, 0xe5c09b7a}},
    {{0x4ce119c9, 0x6e2fa357, 0x200b559b, 0x2f7dd5a5, 0xf02d5290, 0xaff74b03, 0xf3e471b2, 0x73211c97},{0x12ba26dc, 0xb10ec162, 0x5da61fa1, 0x0a844c67, 0x61629482, 0x71d96967, 0x450288ee, 0x9233dc3a}},
    {{0x352bbf4a, 0x4cdd1256, 0x4f93fa33, 0x2ce33330, 0x1d9ad402, 0x71f81071, 0x81340aef, 0x25be59d5},{0x321eb407, 0x5348f534, 0xd59c1825, 0x9dda3e1f, 0x4a1b3b2e, 0x71b1039c, 0x67bd3d8b, 0xcf81998c}},
    {{0x421f5fc9, 0xa2106544, 0x5c96fdb9, 0x1c0c1e2f, 0x2431741c, 0x72713b4b, 0x99ddcb31, 0x6f31e9fc},{0x2b90f16d, 0x11dabdb6, 0x16f6db7e, 0x225d1e14, 0x743034b3, 0x7b223115, 0xdb20717a, 0xd1cd6781}},
    {{0x2fa2104d, 0x6b38d11b, 0x02300105, 0x59879124, 0xe42ab8df, 0xeff5ff29, 0xdc9cdadd, 0x4ecacc3f},{0x02de1068, 0x295dd865, 0xb6456933, 0x5bd5dd80, 0x181d70ec, 0xfc882648, 0x423ba76b, 0x532b7d67}},
    {{0xfe72c435, 0x413d33d4, 0x8ac09c91, 0x61ba8b09, 0x68321543, 0x9d62b794, 0x0502bda8, 0xb202e6ce},{0x6851de06, 0x7ff24a68, 0xd3ab47e0, 0x9d729981, 0x01dc88e3, 0x6b4a9d22, 0x978ed2fb, 0xcf58c5bf}},
    {{0x9248279b, 0x09b4d68d, 0xab21a9b0, 0x66edda83, 0x263c3d84, 0xe09572e2, 0x69ca0cd7, 0xf5453714},{0x73016f7b, 0xf234aade, 0x5d1aa71b, 0xdea2b1ff, 0x3fc0de2a, 0x887912ff, 0xe54a32ce, 0x97cb3402}},
    {{0x6687cdb5, 0xb650d558, 0xf40cbdef, 0xc8e40997, 0xc03fe1b2, 0xabb84088, 0x5e5cad81, 0x710c4c8a},{0x3fd502b3, 0x111178b1, 0x1a1fa873, 0x825c7200, 0x0ef8e529, 0xf033f272, 0xb32e83b2, 0x5c83ad64}},
    {{0xdaed4f2b, 0xe3a8bf27, 0x8e70132f, 0xb0beb752, 0x2f570e14, 0x4bf615c0, 0x7e996d44, 0x3dee8729},{0xa69dce4a, 0x7d6c98e8, 0xd4a1aca8, 0x7ef8d700, 0x3f83c230, 0xf3afa726, 0xab40e522, 0x90be1c55}},
    {{0x55eb67d7, 0xb7238a70, 0xa7fa6f64, 0xd5dc3c82, 0x6b31536d, 0xa6eb344d, 0xc39a66f9, 0x04f97968},{0x7d916a47, 0xb2b58140, 0x0b1e718b, 0xf4042585, 0x40973bce, 0x1c95052d, 0xd0689f2f, 0x493be3c8}},
    {{0xc44d12c7, 0x065d812e, 0x8acf28d7, 0xcbb19f90, 0x11ecd9e9, 0xfdf281b0, 0xe6a3b5e8, 0x7d22e7db},{0x2119a460, 0xce326cdc, 0x76c45926, 0xc982fdac, 0x0e106e86, 0x1edf61c5, 0xa039063f, 0x0e0e6482}},
    {{0x6d2b085e, 0x9e382ed1, 0x0b69fc31, 0x1a03f864, 0x1ccfff21, 0x574de092, 0x7513a49d, 0x9a688a00},{0xacb82eb9, 0x3309ad1c, 0xc739ddfa, 0x33604a83, 0x776238aa, 0x0bd5ff24, 0x8dbac47a, 0x17f388fb}},
    {{0x6a245bf6, 0xdc698504, 0xc89a20cf, 0xded60853, 0x152b6953, 0x36c28063, 0xb61c65cb, 0xd269e6b4},{0xe022cf42, 0xc2bd4a70, 0x8b3f5126, 0xf16a24ad, 0x8b33ba48, 0xd0423b6e, 0xfd5e6348, 0x100d8a82}},
    {{0xd30199d7, 0x4fb5a22d, 0x47b6e054, 0xe2f378ce, 0xdacffcb8, 0x9904a61d, 0x75d0dbd4, 0x07143e65},{0x95038d9d, 0x0ae3d5c3, 0xb3d6dec9, 0xe9838065, 0x1f760cc3, 0x64ed8196, 0x05b3ff1f, 0x24106ab9}}
};
__device__ static _uint256 batch_read(const uint8_t *bytes) {
    uint32_t w[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = ((uint32_t)bytes[4*i] << 24) | ((uint32_t)bytes[4*i+1] << 16)
        | ((uint32_t)bytes[4*i+2] << 8) | bytes[4*i+3];
    return _uint256{w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]};
}
__device__ static void batch_write(_uint256 value, uint8_t *bytes) {
    uint32_t w[8] = {value.a,value.b,value.c,value.d,value.e,value.f,value.g,value.h};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        bytes[4*i] = w[i] >> 24; bytes[4*i+1] = w[i] >> 16;
        bytes[4*i+2] = w[i] >> 8; bytes[4*i+3] = w[i];
    }
}
// Montgomery's batch inversion: one inverse per batch of independent additions.
__device__ void secp256k1_public_add_batch(const uint8_t *in, uint8_t *out) {
    CurvePoint p = {batch_read(in+1),batch_read(in+33)};
    _uint256 products[SECP256K1_BATCH_SIZE];
    _uint256 product = uint32_to_uint256(1);
    for (int i = 0; i < SECP256K1_BATCH_SIZE; ++i) {
        _uint256 dx = sub_256_mod_p(BATCH_G[i].x, p.x);
        if (eqeq_256(dx, uint32_to_uint256(0))) {
            // Degenerate input: preserve the ordinary addition path.
            uint8_t current[65]; memcpy(current,in,65);
            for (int j = 0; j < SECP256K1_BATCH_SIZE; ++j) {
                secp256k1_public_add_generator(current,current);
                memcpy(out+65*j,current,65);
            }
            return;
        }
        product = mul_256_mod_p(product,dx);
        products[i] = product;
    }
    _uint256 inverse = eeuclid_256_mod_p(product);
    for (int i = SECP256K1_BATCH_SIZE - 1; i >= 0; --i) {
        _uint256 dx = sub_256_mod_p(BATCH_G[i].x,p.x);
        _uint256 inv_dx = i ? mul_256_mod_p(inverse,products[i-1]) : inverse;
        inverse = mul_256_mod_p(inverse,dx);
        _uint256 lambda = mul_256_mod_p(sub_256_mod_p(BATCH_G[i].y,p.y),inv_dx);
        _uint256 qx = sub_256_mod_p(sub_256_mod_p(mul_256_mod_p(lambda,lambda),p.x),BATCH_G[i].x);
        _uint256 qy = sub_256_mod_p(mul_256_mod_p(lambda,sub_256_mod_p(p.x,qx)),p.y);
        out[65*i] = 4;
        batch_write(qx,out+65*i+1); batch_write(qy,out+65*i+33);
    }
}
