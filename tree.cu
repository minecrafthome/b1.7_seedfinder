//"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc.exe"  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x64" -o do do.cu -O3 -m=64 -arch=compute_61 -code=sm_61 -Xptxas -allow-expensive-optimizations=true
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>


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

__host__ __device__ static inline void advance6(Random *random) {
    *random = (*random * 0x45D73749A7F9ULL + 0x17617168255EULL) & RANDOM_MASK;
}

__host__ __device__ static inline void advance69(Random *random) {
    *random = (*random * 0x48EF66D8C53DULL + 0xAD1D21AF87FFULL) & RANDOM_MASK;
}

__host__ __device__ static inline void advance3780(Random *random) {
    *random = (*random * 0xF7D729EDC211ULL + 0x140CD08661C4ULL) & RANDOM_MASK;
}

__constant__ uint64_t const lake_multipliers[4] = {0x8BEF0ACF36C1ULL, 0xAA5EABD3FD81ULL, 0x585E479A5441ULL, 0x62F233AE3B01ULL};
__constant__ uint64_t const lake_addends[4] = {0x7F29273874B0ULL, 0x76FADBB58D60ULL, 0xF894278A4A10ULL, 0x5BD8A509AAC0ULL};

#endif //TERRAINGENCPP_JAVARND_H




































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













#define BLOCK_SIZE_BITS 9
#define WORK_SIZE_BITS 14
#define SEEDS_PER_CALL (1ULL << (BLOCK_SIZE_BITS + WORK_SIZE_BITS))

__device__ static inline bool checkTree(Random* rng) {
	int x = random_next(rng, 4);
	int z = random_next(rng, 4);
	if (random_next_int(rng, 10) == 0) {
		// big tree
		advance2(rng);
	} else {
		// smol tree
		int height_val = random_next_int(rng, 3);
		if (x == 7 && z == 6 && height_val != 2) {
			bool result = true;
			random_next(rng, 1);
			result &= random_next(rng, 1) != 0;
			advance2(rng);
			result &= random_next(rng, 1) != 0;
			result &= random_next(rng, 1) != 0;
			result &= random_next(rng, 1) != 0;
			result &= random_next(rng, 1) != 0;
			result &= random_next(rng, 1) == 0;
			result &= random_next(rng, 1) == 0;
			advance6(rng);
			return result;
		}
	}
	return false;
}


__device__ static inline bool checkTrees(Random rng) {
	return checkTree(&rng) || checkTree(&rng) || checkTree(&rng) || checkTree(&rng) || checkTree(&rng) || checkTree(&rng) || checkTree(&rng) || checkTree(&rng) || checkTree(&rng) || checkTree(&rng);
}

__global__ __launch_bounds__(1ULL<<BLOCK_SIZE_BITS,4) static void checkSeed(uint64_t worldSeedOffset, uint64_t* output, uint32_t* count) {
	uint64_t seed = blockIdx.x * blockDim.x + threadIdx.x + worldSeedOffset;
	
	int64_t chunkX = 4;
	int64_t chunkZ = -5;
	register Random rng = get_random(seed);
	int64_t l1 = (((int64_t)random_next_long(&rng)) / 2LL) * 2LL + 1LL;
    int64_t l2 = (((int64_t)random_next_long(&rng)) / 2LL) * 2LL + 1LL;
	
    rng = get_random((int64_t)chunkX * l1 + (int64_t)chunkZ * l2 ^ seed);

	// water lakes
	if (random_next(&rng, 2) == 0) {
		advance3(&rng);
		int rand_value = random_next(&rng, 2);
		rng = (rng * __ldg(&lake_multipliers[rand_value]) + __ldg(&lake_addends[rand_value])) & 0xffffffffffffULL;
	}
	
	// lava lakes
	if (random_next(&rng, 3) == 0) {
		random_next(&rng, 1);
		int y = random_next_int(&rng, random_next_int(&rng, 120) + 8);
		random_next(&rng, 1);
		if (y < 64 || random_next_int(&rng, 10) == 0) {
			int rand_value = random_next(&rng, 2);
			rng = (rng * __ldg(&lake_multipliers[rand_value]) + __ldg(&lake_addends[rand_value])) & 0xffffffffffffULL;
		}
	}

	// ores and stuff
	advance3780(&rng);

	// advance69 accounts for the case of a clay patch
	if (checkTrees(rng)) {// || (advance69(&rng), checkTrees(rng))
		uint32_t index = atomicAdd(count, 1);
		output[index] = seed;
	}
}















void process_seeds(uint32_t count, uint64_t* seeds) {

}





























#include <windows.h>
int main() {
	printf("doing\n");
	
	cudaSetDevice(0);
	uint64_t* buffer;
	cudaMallocManaged(&buffer, sizeof(*buffer) * SEEDS_PER_CALL);
	uint32_t* count;
	cudaMallocManaged(&count, sizeof(*count));
	//testMem<<<1,1>>>(0);
	//cudaDeviceSynchronize();
	//return 0;
	for (uint64_t seed =0; seed< (281474976710656ULL);seed+=SEEDS_PER_CALL) {
		SYSTEMTIME time;
		GetSystemTime(&time);
		LONG start = (time.wSecond * 1000) + time.wMilliseconds;
		
		*count = 0;
		checkSeed<<< 1ULL << WORK_SIZE_BITS, 1ULL << BLOCK_SIZE_BITS >>>(seed, buffer, count);
   
		cudaDeviceSynchronize();
		printf("%I64u\n", *count);
		*count = 0;
		
		
		GetSystemTime(&time);
		LONG end = (time.wSecond * 1000) + time.wMilliseconds;
		printf("Time elapsed %dms, speed: %.2fm/s\n", (int)(end - start),((double)((1ULL<<WORK_SIZE_BITS)*(1ULL<<BLOCK_SIZE_BITS)))/((double)(end - start))/1000.0);		
	}
}
















