

// Inside GPU, use __half and __nv_bfloat16 types and the corresponding
// intrinsics for conversion and math operations.

#pragma once
#include <cstdint>
#include <cstring>

// bfloat16 bits -> float32
static inline float bf16_to_float(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float out;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

// it is okay to use fp16 but,
// before using fp16 ask yourself why not fp32 or bf16 ?!?!
// IEEE fp16 (binary16) bits -> float32
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t expu =
        (uint32_t)(h & 0x7C00u) >> 10; // keep original exponent as unsigned
    uint32_t mant = (uint32_t)(h & 0x03FFu);

    uint32_t f;
    if (expu == 0) {
        if (mant == 0) {
            f = sign; // zero
        } else {
            int32_t exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;

            uint32_t exp_f = (uint32_t)(exp + (127 - 15));
            uint32_t mant_f = mant << 13;
            f = sign | (exp_f << 23) | mant_f;
        }
    } else if (expu == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);

    } else {
        uint32_t exp_f = expu + (127 - 15);
        uint32_t mant_f = mant << 13;
        f = sign | (exp_f << 23) | mant_f;
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}
