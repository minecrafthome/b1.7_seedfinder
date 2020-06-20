//"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc.exe"  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x64" -o crack crack.cu -O3 -m=64 -arch=compute_61 -code=sm_61 -Xptxas -allow-expensive-optimizations=true
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>



// ===== LCG IMPLEMENTATION ===== //

#ifndef TERRAINGENCPP_JAVARND_H
#define TERRAINGENCPP_JAVARND_H
#define Random uint64_t
#define RANDOM_MULTIPLIER 0x5DEECE66DULL
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK ((1ULL << 48u) - 1)
#define get_random(seed) ((Random)((seed ^ RANDOM_MULTIPLIER) & RANDOM_MASK))


__host__ __device__ static inline int32_t random_next(Random *random, int bits) {
    *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
    return (int32_t) (*random >> (48u - bits));
}

__host__ __device__ static inline int32_t random_next_int(Random *random, const uint16_t bound) {
    int32_t r = random_next(random, 31);
    const uint16_t m = bound - 1u;
    if ((bound & m) == 0) {
        r = (int32_t) ((bound * (uint64_t) r) >> 31u);
    } else {
        for (int32_t u = r;
             u - (r = u % bound) + m < 0;
             u = random_next(random, 31));
    }
    return r;
}

__host__ __device__ static inline double next_double(Random *random) {
    return (double) ((((uint64_t) ((uint32_t) random_next(random, 26)) << 27u)) + random_next(random, 27)) / (double)(1ULL << 53);
}
__host__ __device__ static inline uint64_t random_next_long (Random *random) {
    return (((uint64_t)random_next(random, 32)) << 32u) + (int32_t)random_next(random, 32);
}
__host__ __device__ static inline void advance2(Random *random) {
    *random = (*random * 0xBB20B4600A69LLU + 0x40942DE6BALLU) & RANDOM_MASK;
}

__host__ __device__ static inline void advance3(Random *random) {
    *random = (*random * 0xD498BD0AC4B5ULL + 0xAA8544E593DLL) & RANDOM_MASK;
}

__host__ __device__ static inline void advance4(Random *random) {
    *random = (*random * 0x32EB772C5F11ULL + 0x2D3873C4CD04ULL) & RANDOM_MASK;
}

__host__ __device__ static inline void advance69(Random *random) {
    *random = (*random * 0x48EF66D8C53DULL + 0xAD1D21AF87FFULL) & RANDOM_MASK;
}

__host__ __device__ static inline void advance3780(Random *random) {
    *random = (*random * 0xF7D729EDC211ULL + 0x140CD08661C4ULL) & RANDOM_MASK;
}

__host__ __device__ static inline void advance3849(Random *random) {
    *random = (*random * 0xB3567856530DULL + 0x32F4D050A7B3ULL) & RANDOM_MASK;
}

__constant__ uint64_t const lake_multipliers[4] = {0x8BEF0ACF36C1ULL, 0xAA5EABD3FD81ULL, 0x585E479A5441ULL, 0x62F233AE3B01ULL};
__constant__ uint64_t const lake_addends[4] = {0x7F29273874B0ULL, 0x76FADBB58D60ULL, 0xF894278A4A10ULL, 0x5BD8A509AAC0ULL};

#endif //TERRAINGENCPP_JAVARND_H



// ===== DEVICE INTRINSICS ===== //

#define DEVICE_STATIC_INTRINSIC_QUALIFIERS  static __device__ __forceinline__

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define PXL_GLOBAL_PTR   "l"
#else
#define PXL_GLOBAL_PTR   "r"
#endif

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_local_l1(const void* const ptr)
{
  asm("prefetch.local.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_uniform(const void* const ptr)
{
  asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_local_l2(const void* const ptr)
{
  asm("prefetch.local.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

#if __CUDA__ < 10
#define __ldg(ptr) (*(ptr))
#endif



// ===== BIOME GENERATIOR IMPLEMENTATION ===== //

__constant__ uint8_t const biomeLookup[] = {11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1};


struct SimplexOctave {
	double xo;
    double yo;
    uint8_t permutations[256];
};


#define F2 0.3660254037844386
#define G2 0.21132486540518713

__constant__ __device__ int8_t const grad2[12][2] = {{1,  1,},
                    {-1, 1,},
                    {1,  -1,},
                    {-1, -1,},
                    {1,  0,},
                    {-1, 0,},
                    {1,  0,},
                    {-1, 0,},
                    {0,  1,},
                    {0,  -1,},
                    {0,  1,},
                    {0,  -1,}};

/* End of constant for simplex noise*/

#define getValue(array, index) array[index]

/* simplex noise result is in buffer */
__device__ static inline double getSimplexNoise(double chunkX, double chunkZ, double offsetX, double offsetZ, double ampFactor, uint8_t nbOctaves, SimplexOctave octaves[], Random *random) {
    offsetX /= 1.5;
    offsetZ /= 1.5;
    double res = 0.0;
    double octaveDiminution = 1.0;
    double octaveAmplification = 1.0;
    for (uint8_t j = 0; j < nbOctaves; ++j) {
		SimplexOctave* oct = &octaves[j];
        oct->xo = next_double(random) * 256.0;
        oct->yo = next_double(random) * 256.0;
		uint8_t* permutations = oct->permutations; 
        advance2(random);
		#pragma unroll
        for(uint16_t w = 0; w<256; w++) {
			permutations[w] = w;
			__prefetch_local_l2(permutations+(w));
		}
        
        for(uint16_t index = 0; index<256; index++) {
			__prefetch_local_l1(permutations+((index + 1)));
			//__prefetch_global_l1(permutations+index);
            uint32_t randomIndex = random_next_int(random, 256ull - index) + index;
            if (randomIndex != index) {
                // swap
				uint8_t v1 = permutations[index];
				uint8_t v2 = permutations[randomIndex];
				
                permutations[index] = v2;
                permutations[randomIndex] = v1;
            }
        }
        double XCoords = (double) chunkX * offsetX * octaveAmplification + oct->xo;
        double ZCoords = (double) chunkZ * offsetZ * octaveAmplification + oct->yo;
        // Skew the input space to determine which simplex cell we're in
        double hairyFactor = (XCoords + ZCoords) * F2;
        auto tempX = static_cast<int32_t>(XCoords + hairyFactor);
        auto tempZ = static_cast<int32_t>(ZCoords + hairyFactor);
        int32_t xHairy = (XCoords + hairyFactor < tempX) ? (tempX - 1) : (tempX);
        int32_t zHairy = (ZCoords + hairyFactor < tempZ) ? (tempZ - 1) : (tempZ);
		// Work out the hashed gradient indices of the three simplex corners
        uint8_t ii = (uint32_t) xHairy & 0xffu;
        uint8_t jj = (uint32_t) zHairy & 0xffu;
		__prefetch_local_l1(permutations + (jj));
		__prefetch_local_l1(permutations+((jj+1)));
		//__prefetch_local_l1(&permutations[(uint16_t)(jj + 1)& 0xffu]);
		
        double d11 = (double) (xHairy + zHairy) * G2;
        double X0 = (double) xHairy - d11; // Unskew the cell origin back to (x,y) space
        double Y0 = (double) zHairy - d11;
        double x0 = XCoords - X0; // The x,y distances from the cell origin
        double y0 = ZCoords - Y0;
        // For the 2D case, the simplex shape is an equilateral triangle.
        // Determine which simplex we are in.
        int offsetSecondCornerX, offsetSecondCornerZ; // Offsets for second (middle) corner of simplex in (i,j) coords

        if (x0 > y0) {  // lower triangle, XY order: (0,0)->(1,0)->(1,1)
            offsetSecondCornerX = 1;
            offsetSecondCornerZ = 0;
        } else { // upper triangle, YX order: (0,0)->(0,1)->(1,1)
            offsetSecondCornerX = 0;
            offsetSecondCornerZ = 1;
        }

        double x1 = (x0 - (double) offsetSecondCornerX) + G2; // Offsets for middle corner in (x,y) unskewed coords
        double y1 = (y0 - (double) offsetSecondCornerZ) + G2;
        double x2 = (x0 - 1.0) + 2.0 * G2; // Offsets for last corner in (x,y) unskewed coords
        double y2 = (y0 - 1.0) + 2.0 * G2;

        
        uint8_t gi0 = getValue(permutations,(uint16_t) (ii + getValue(permutations,jj)) & 0xffu) % 12u;
        uint8_t gi1 = getValue(permutations,(uint16_t)(ii + offsetSecondCornerX + getValue(permutations,(uint16_t) (jj + offsetSecondCornerZ) & 0xffu))& 0xffu) % 12u;
        uint8_t gi2 = getValue(permutations,(uint16_t)(ii + 1 + getValue(permutations,(uint16_t)(jj + 1)& 0xffu))& 0xffu) % 12u;

        // Calculate the contribution from the three corners
        double t0 = 0.5 - x0 * x0 - y0 * y0;
        double n0;
        if (t0 < 0.0) {
            n0 = 0.0;
        } else {
            t0 *= t0;
            n0 = t0 * t0 * ((double) __ldg(&grad2[gi0][0]) * x0 + (double) __ldg(&grad2[gi0][1]) * y0);  // (x,y) of grad2 used for 2D gradient
        }
        double t1 = 0.5 - x1 * x1 - y1 * y1;
        double n1;
        if (t1 < 0.0) {
            n1 = 0.0;
        } else {
            t1 *= t1;
            n1 = t1 * t1 * ((double) __ldg(&grad2[gi1][0]) * x1 + (double) __ldg(&grad2[gi1][1]) * y1);
        }
        double t2 = 0.5 - x2 * x2 - y2 * y2;
        double n2;
        if (t2 < 0.0) {
            n2 = 0.0;
        } else {
            t2 *= t2;
            n2 = t2 * t2 * ((double) __ldg(&grad2[gi2][0]) * x2 + (double) __ldg(&grad2[gi2][1]) * y2);
        }
        // Add contributions from each corner to get the final noise value.
        // The result is scaled to return values in the interval [-1,1].
        res = res + 70.0 * (n0 + n1 + n2) * 0.55000000000000004 / octaveDiminution;
        octaveAmplification *= ampFactor;
        octaveDiminution *= 0.5;
    }
    return res;

}

__device__ static inline double getSimplexNoiseFromOctives(double chunkX, double chunkZ, double offsetX, double offsetZ, double ampFactor, uint8_t nbOctaves, SimplexOctave octaves[]) {
    offsetX /= 1.5;
    offsetZ /= 1.5;
    double res = 0.0;
    double octaveDiminution = 1.0;
    double octaveAmplification = 1.0;
    for (uint8_t j = 0; j < nbOctaves; ++j) {
		SimplexOctave* oct = &octaves[j];
		uint8_t* permutations = oct->permutations; 
        double XCoords = (double) chunkX * offsetX * octaveAmplification + oct->xo;
        double ZCoords = (double) chunkZ * offsetZ * octaveAmplification + oct->yo;
        // Skew the input space to determine which simplex cell we're in
        double hairyFactor = (XCoords + ZCoords) * F2;
        auto tempX = static_cast<int32_t>(XCoords + hairyFactor);
        auto tempZ = static_cast<int32_t>(ZCoords + hairyFactor);
        int32_t xHairy = (XCoords + hairyFactor < tempX) ? (tempX - 1) : (tempX);
        int32_t zHairy = (ZCoords + hairyFactor < tempZ) ? (tempZ - 1) : (tempZ);
		// Work out the hashed gradient indices of the three simplex corners
        uint8_t ii = (uint32_t) xHairy & 0xffu;
        uint8_t jj = (uint32_t) zHairy & 0xffu;
		__prefetch_local_l1(permutations + (jj));
		__prefetch_local_l1(permutations+((jj+1)));
		//__prefetch_local_l1(&permutations[(uint16_t)(jj + 1)& 0xffu]);
		
        double d11 = (double) (xHairy + zHairy) * G2;
        double X0 = (double) xHairy - d11; // Unskew the cell origin back to (x,y) space
        double Y0 = (double) zHairy - d11;
        double x0 = XCoords - X0; // The x,y distances from the cell origin
        double y0 = ZCoords - Y0;
        // For the 2D case, the simplex shape is an equilateral triangle.
        // Determine which simplex we are in.
        int offsetSecondCornerX, offsetSecondCornerZ; // Offsets for second (middle) corner of simplex in (i,j) coords

        if (x0 > y0) {  // lower triangle, XY order: (0,0)->(1,0)->(1,1)
            offsetSecondCornerX = 1;
            offsetSecondCornerZ = 0;
        } else { // upper triangle, YX order: (0,0)->(0,1)->(1,1)
            offsetSecondCornerX = 0;
            offsetSecondCornerZ = 1;
        }

        double x1 = (x0 - (double) offsetSecondCornerX) + G2; // Offsets for middle corner in (x,y) unskewed coords
        double y1 = (y0 - (double) offsetSecondCornerZ) + G2;
        double x2 = (x0 - 1.0) + 2.0 * G2; // Offsets for last corner in (x,y) unskewed coords
        double y2 = (y0 - 1.0) + 2.0 * G2;

        
        uint8_t gi0 = getValue(permutations,(uint16_t) (ii + getValue(permutations,jj)) & 0xffu) % 12u;
        uint8_t gi1 = getValue(permutations,(uint16_t)(ii + offsetSecondCornerX + getValue(permutations,(uint16_t) (jj + offsetSecondCornerZ) & 0xffu))& 0xffu) % 12u;
        uint8_t gi2 = getValue(permutations,(uint16_t)(ii + 1 + getValue(permutations,(uint16_t)(jj + 1)& 0xffu))& 0xffu) % 12u;

        // Calculate the contribution from the three corners
        double t0 = 0.5 - x0 * x0 - y0 * y0;
        double n0;
        if (t0 < 0.0) {
            n0 = 0.0;
        } else {
            t0 *= t0;
            n0 = t0 * t0 * ((double) __ldg(&grad2[gi0][0]) * x0 + (double) __ldg(&grad2[gi0][1]) * y0);  // (x,y) of grad2 used for 2D gradient
        }
        double t1 = 0.5 - x1 * x1 - y1 * y1;
        double n1;
        if (t1 < 0.0) {
            n1 = 0.0;
        } else {
            t1 *= t1;
            n1 = t1 * t1 * ((double) __ldg(&grad2[gi1][0]) * x1 + (double) __ldg(&grad2[gi1][1]) * y1);
        }
        double t2 = 0.5 - x2 * x2 - y2 * y2;
        double n2;
        if (t2 < 0.0) {
            n2 = 0.0;
        } else {
            t2 *= t2;
            n2 = t2 * t2 * ((double) __ldg(&grad2[gi2][0]) * x2 + (double) __ldg(&grad2[gi2][1]) * y2);
        }
        // Add contributions from each corner to get the final noise value.
        // The result is scaled to return values in the interval [-1,1].
        res = res + 70.0 * (n0 + n1 + n2) * 0.55000000000000004 / octaveDiminution;
        octaveAmplification *= ampFactor;
        octaveDiminution *= 0.5;
    }
    return res;

}


__device__ static inline double getTempFromTempAndPrecip(double temp, double precip) {
	precip = precip  * 1.1000000000000001 + 0.5;
	temp = (temp * 0.14999999999999999 + 0.69999999999999996) * (1.0 - 0.01) + precip * 0.01;
	
	temp = 1.0 - (1.0 - temp) * (1.0 - temp);
	if (temp < 0.0) {
		temp = 0.0;
	}
	if (temp > 1.0) {
		temp = 1.0;
	}
	return temp;
}

__device__ static inline double getHumidFromHumidAndPrecip(double humidity, double precip) {
	precip = precip  * 1.1000000000000001 + 0.5;
	humidity = (humidity * 0.14999999999999999 + 0.5) * (1.0 - 0.002) + precip * 0.002;
	if (humidity < 0.0) {
		humidity = 0.0;
	}
	if (humidity > 1.0) {
		humidity = 1.0;
	}
	return humidity;
}


#define ConvertToIndex(value) ((int32_t)((value)*63.0))

__device__ static inline uint8_t getBiome(int x, int z, SimplexOctave precipOctaves[], SimplexOctave tempOctaves[], SimplexOctave humidOctaves[]) {
	double precipAtPos =  getSimplexNoiseFromOctives((double)x, (double)z, 0.25, 0.25, 0.58823529411764708, 2, precipOctaves);
	double tempAtPos = getSimplexNoiseFromOctives((double)x, (double)z, 0.02500000037252903, 0.02500000037252903, 0.25, 4, tempOctaves);
	double humidityAtPos = getSimplexNoiseFromOctives((double)x, (double)z, 0.05000000074505806, 0.05000000074505806, 0.33333333333333331, 4, humidOctaves);
	int32_t index = ConvertToIndex(getTempFromTempAndPrecip(tempAtPos, precipAtPos)) + ConvertToIndex(getHumidFromHumidAndPrecip(humidityAtPos,precipAtPos)) * 64;
	return __ldg(&biomeLookup[index]);
}



// ===== THE BRUTEFORCE ===== //

//BLOCK_SIZE_BITS-1),2
#define BLOCK_SIZE_BITS 8
#define WORK_SIZE_BITS 23
#define SEEDS_PER_CALL (1ULL << (BLOCK_SIZE_BITS + WORK_SIZE_BITS))

#define PLAINS_BIOME_X 48
#define PLAINS_BIOME_Z -72
#define DESERT_BIOME_X 47
#define DESERT_BIOME_Z -72

//DOUBLE CHECK WITH EARTH
#define SEASONAL_FOREST_BIOME_X  80
#define SEASONAL_FOREST_BIOME_Z  -64


__global__ __launch_bounds__(1ULL<<(BLOCK_SIZE_BITS-1),2) static void checkSeedBiomes(uint64_t* bothput, uint32_t count) {

	uint32_t seedIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (seedIndex >= count)
		return;
	int64_t seed = (int64_t)bothput[seedIndex];
	
		
	register Random biomeSeed = get_random(seed * 0x84a59LL);
	SimplexOctave precipOctaves[2];
	double precipAtPos = getSimplexNoise((double)PLAINS_BIOME_X, (double)PLAINS_BIOME_Z, 0.25, 0.25, 0.58823529411764708, 2, precipOctaves, &biomeSeed);//0.0, 0.0 are the block coordinates
	
	
	biomeSeed = get_random(seed  * 9871LL);
	SimplexOctave tempOctaves[4];
	double tempAtPos = getSimplexNoise((double)PLAINS_BIOME_X, (double)PLAINS_BIOME_Z, 0.02500000037252903, 0.02500000037252903, 0.25, 4, tempOctaves, &biomeSeed);
	
	
	//Double check its correct, checking if its a plains biome
	if (ConvertToIndex(getTempFromTempAndPrecip(tempAtPos, precipAtPos))<62) {
		bothput[seedIndex] = 0;
		return;
	}
	
	
	biomeSeed = get_random(seed  * 39811LL);
	SimplexOctave humidOctaves[4];
	double humidityAtPos = getSimplexNoise((double)PLAINS_BIOME_X, (double)PLAINS_BIOME_Z, 0.05000000074505806, 0.05000000074505806, 0.33333333333333331, 4, humidOctaves, &biomeSeed);
	
	
	int32_t humid = ConvertToIndex(getHumidFromHumidAndPrecip(humidityAtPos,precipAtPos));
	//Double check its correct, checking if its a plains biome
	if ((humid<14) || (humid>29)){
		bothput[seedIndex] = 0;
		return;
	}
	
	
	precipAtPos = getSimplexNoiseFromOctives((double)DESERT_BIOME_X, (double)DESERT_BIOME_Z, 0.25, 0.25, 0.58823529411764708, 2, precipOctaves);
	tempAtPos = getSimplexNoiseFromOctives((double)DESERT_BIOME_X, (double)DESERT_BIOME_Z, 0.02500000037252903, 0.02500000037252903, 0.25, 4, tempOctaves);
	if (ConvertToIndex(getTempFromTempAndPrecip(tempAtPos, precipAtPos))<60) {
		bothput[seedIndex] = 0;
		return;
	}
	
	humidityAtPos = getSimplexNoiseFromOctives((double)DESERT_BIOME_X, (double)DESERT_BIOME_Z, 0.05000000074505806, 0.05000000074505806, 0.33333333333333331, 4, humidOctaves);
	int32_t index = ConvertToIndex(getTempFromTempAndPrecip(tempAtPos, precipAtPos)) + ConvertToIndex(getHumidFromHumidAndPrecip(humidityAtPos,precipAtPos)) * 64;
	if (__ldg(&biomeLookup[index]) != 8){
		bothput[seedIndex] = 0;
		return;
	}
	
	
	//If its not a seasonalForrest
	if (getBiome(SEASONAL_FOREST_BIOME_X, SEASONAL_FOREST_BIOME_Z, precipOctaves, tempOctaves, humidOctaves)!=3) {
		bothput[seedIndex] = 0;
		return;
	}
	
}

#define PRSTR(it) if (false) printf(it "\n")
#define PRRND(it) if (false) printf("%lld\n", (it))

__device__ static inline bool checkTree(uint64_t seed, Random* rng) {
	PRSTR("Checking tree");
	PRRND(*rng);
	int x = random_next(rng, 4);
	int z = random_next(rng, 4);
	if (random_next_int(rng, 10) == 0) {
		PRSTR("Big tree");
		// big tree
		advance2(rng);
	} else {
		PRSTR("Smol tree");
		// smol tree
		int height_val = random_next_int(rng, 3);
		PRRND(*rng);
		PRRND(x);
		PRRND(z);
		PRRND(height_val);
		if (x == 7 && z == 6 && height_val != 2) {
			bool result = true;
			random_next(rng, 1);
			result &= random_next(rng, 1) != 0;
			advance2(rng);
			result &= random_next(rng, 1) != 0;
			result &= random_next(rng, 1) != 0;
			result &= random_next(rng, 1) != 0;
			result &= random_next(rng, 1) == 0;
			result &= random_next(rng, 1) == 0;
			result &= random_next(rng, 1) == 0;
			random_next(rng, 1);
			result &= random_next(rng, 1) != 0;
			advance4(rng);
			if (result) {
				PRSTR("true");
			} else {
				PRSTR("false");
			}
			return result;
		}
	}
	return false;
}


__device__ static inline bool checkTrees(uint64_t seed, Random rng) {
	return checkTree(seed, &rng) || checkTree(seed, &rng) || checkTree(seed, &rng) || checkTree(seed, &rng) || checkTree(seed, &rng) || checkTree(seed, &rng) || checkTree(seed, &rng) || checkTree(seed, &rng) || checkTree(seed, &rng) || checkTree(seed, &rng);
}


__global__ __launch_bounds__(1ULL<<BLOCK_SIZE_BITS,4) static void checkSeed(uint64_t worldSeedOffset, uint64_t* output, uint32_t* count) {
	uint64_t seed = blockIdx.x * blockDim.x + threadIdx.x + worldSeedOffset;
	int64_t chunkX = 4;
	int64_t chunkZ = -5;
	register Random rng = get_random(seed);
	int64_t l1 = (((int64_t)random_next_long(&rng)) / 2LL) * 2LL + 1LL;
    int64_t l2 = (((int64_t)random_next_long(&rng)) / 2LL) * 2LL + 1LL;
	
    rng = get_random((int64_t)chunkX * l1 + (int64_t)chunkZ * l2 ^ seed);
	PRRND(rng);

	// water lakes
	if (random_next(&rng, 2) == 0) {
		PRSTR("Water lakes");
		PRRND(rng);
		advance3(&rng);
		int rand_value = random_next(&rng, 2);
		rng = (rng * __ldg(&lake_multipliers[rand_value]) + __ldg(&lake_addends[rand_value])) & 0xffffffffffffULL;
		PRRND(rng);
	}
	
	// lava lakes
	if (random_next(&rng, 3) == 0) {
		PRSTR("Lava lakes");
		PRRND(rng);
		random_next(&rng, 1);
		int y = random_next_int(&rng, random_next_int(&rng, 120) + 8);
		random_next(&rng, 1);
		if (y < 64 || random_next_int(&rng, 10) == 0) {
			PRSTR("Lava lake actually being attempted");
			PRRND(rng);
			int rand_value = random_next(&rng, 2);
			rng = (rng * __ldg(&lake_multipliers[rand_value]) + __ldg(&lake_addends[rand_value])) & 0xffffffffffffULL;
		}
		PRRND(rng);
	}

	// ores and stuff
	//advance3780(&rng);
	// advance69 accounts for the case of a clay patch
	//advance69(&rng);
	// combine ores and clay
	advance3849(&rng);
	PRRND(rng);

	if (checkTrees(seed, rng)) {
		uint32_t index = atomicAdd(count, 1);
		output[index] = seed;
	}
}


#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
    exit(code);
  }
}


#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <windows.h>
	uint64_t getCurrentTimeMillis() {
		SYSTEMTIME time;
		GetSystemTime(&time);
		return (uint64_t)((time.wSecond * 1000) + time.wMilliseconds);
	}
#else
	#include <sys/time.h>
	uint64_t getCurrentTimeMillis() {
		struct timeval te; 
		gettimeofday(&te, NULL); // get current time
		uint64_t milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // calculate milliseconds
		return milliseconds;
	}
#endif


uint64_t actual_count = 0;
int main(int argc, char** argv) {
	if (argc < 3) {
		fprintf(stderr, "%s <from_batch_inclusive> <to_batch_exclusive> [gpu_device]\n", argv[0]);
		return 0;
	}
	int start_batch = atoi(argv[1]);
	int end_batch = atoi(argv[2]);
	if (start_batch < 0 || start_batch >= end_batch || end_batch > (1ULL << 48) / SEEDS_PER_CALL) {
		fprintf(stderr, "Invalid batch bounds\n");
		return 0;
	}
	int gpu_device = argc <= 3 ? 0 : atoi(argv[3]);

	fprintf(stderr, "doing between %lld (inclusive) and %lld (exclusive)\n", start_batch * SEEDS_PER_CALL, end_batch * SEEDS_PER_CALL);
	FILE* out_file = fopen("./seed_output.dat","wb");
	
	
	cudaSetDevice(gpu_device);
	uint64_t* buffer;
	GPU_ASSERT(cudaMallocManaged(&buffer, sizeof(*buffer) * (SEEDS_PER_CALL>>5)));
	GPU_ASSERT(cudaPeekAtLastError());
	uint32_t* count;
	GPU_ASSERT(cudaMallocManaged(&count, sizeof(*count)));
	GPU_ASSERT(cudaPeekAtLastError());
	for (uint64_t seed = start_batch * SEEDS_PER_CALL, end_seed = end_batch * SEEDS_PER_CALL; seed < end_seed; seed+=SEEDS_PER_CALL) {
		uint64_t start = getCurrentTimeMillis();
		
		*count = 0;
		checkSeed<<< 1ULL << WORK_SIZE_BITS, 1ULL << BLOCK_SIZE_BITS >>>(seed, buffer, count);
		GPU_ASSERT(cudaPeekAtLastError());
		GPU_ASSERT(cudaDeviceSynchronize());
		checkSeedBiomes<<< 1ULL << WORK_SIZE_BITS, 1ULL << (BLOCK_SIZE_BITS-2) >>>(buffer, *count);
		GPU_ASSERT(cudaPeekAtLastError());
		GPU_ASSERT(cudaDeviceSynchronize());
		if ((*count - 100) >(SEEDS_PER_CALL>>5)) {
			fprintf(stderr,"MEGA PANNIC, the resulting seed count was bigger than the seed buffer.");
			return -2;
		}
		for(uint64_t i =0;i<*count;i++) {
			if (buffer[i]!=0) {
				actual_count++;
				fwrite(&buffer[i],sizeof(uint64_t),1,out_file);
				fflush(out_file);
				fprintf(stderr, "%llu\n",buffer[i]);
				//return 0;
			}
		}
		
		uint64_t end = getCurrentTimeMillis();
		fprintf(stderr, "Time elapsed %dms, speed: %.2fm/s, seed count: %llu, percent done: %f\n", (int)(end - start),((double)((1ULL<<WORK_SIZE_BITS)*(1ULL<<BLOCK_SIZE_BITS)))/((double)(end - start))/1000.0, actual_count,(((double)seed)/(1ULL<<48))*100);		
	}
	fclose(out_file);
	fprintf(stderr, "Finished work unit\n");
	return 0;
}
