/*
 * secp256k1 implementation based on MrSpike63's proven working vanity-eth-address
 * Uses affine coordinates (simpler than Jacobian)
 * Structure adapted from their GPU implementation with PTX assembly math
 */

#include "secp256k1.h"
#include <stdint.h>

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
