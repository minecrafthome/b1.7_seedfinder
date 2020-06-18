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




#ifndef SIMPLEX_NOISE_H
#define SIMPLEX_NOISE_H
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

#define getValueFromArrayVal(val, index) (val&(0xFFULL<<(((index)&0x1)<<3)))
#define getValue(array, index) getValueFromArrayVal(array[index>>1],index)

/* simplex noise result is in buffer */
__device__ static inline double getSimplexNoise(double chunkX, double chunkZ, double offsetX, double offsetZ, double ampFactor, uint8_t nbOctaves, Random *random) {
    offsetX /= 1.5;
    offsetZ /= 1.5;
    double res = 0.0;
    double octaveDiminution = 1.0;
    double octaveAmplification = 1.0;
    double xo;
    double yo;
    uint16_t permutations[128];//
    for (uint8_t j = 0; j < nbOctaves; ++j) {
        xo = next_double(random) * 256.0;
        yo = next_double(random) * 256.0;
        advance2(random);
		#pragma unroll
        for(uint16_t w = 0; w<256; w+=2) {
			permutations[w>>1] = w|((w+1)<<8);//|((w+1)<<16)|((w+1)<<24);
			__prefetch_local_l2(permutations+(w>>1));
		}
        
        for(uint16_t index = 0; index<256; index++) {
			__prefetch_local_l1(permutations+((index + 1)>>1));
			//__prefetch_global_l1(permutations+index);
            uint32_t randomIndex = random_next_int(random, 256u - index) + index;
            if (randomIndex != index) {
                // swap
				uint16_t v1 = permutations[index>>1];
				uint8_t t1 = getValueFromArrayVal(v1, index);
				
				uint16_t v2 = permutations[randomIndex>>1];
                uint8_t t2 = getValueFromArrayVal(v2, randomIndex);
				
				v1 = (v1&(~((uint16_t)(0xFFULL<<(((index)&0x1)<<3)))))  |  (((uint16_t)t2)<<(((index)&0x1)<<3));
				v2 = (v2&(~((uint16_t)(0xFFULL<<(((randomIndex)&0x1)<<3)))))  |  (((uint16_t)t1)<<(((randomIndex)&0x1)<<3));
				
                permutations[index>>1] = v1;
                permutations[randomIndex>>1] = v2;
            }
        }
        double XCoords = (double) chunkX * offsetX * octaveAmplification + xo;
        double ZCoords = (double) chunkZ * offsetZ * octaveAmplification + yo;
        // Skew the input space to determine which simplex cell we're in
        double hairyFactor = (XCoords + ZCoords) * F2;
        auto tempX = static_cast<int32_t>(XCoords + hairyFactor);
        auto tempZ = static_cast<int32_t>(ZCoords + hairyFactor);
        int32_t xHairy = (XCoords + hairyFactor < tempX) ? (tempX - 1) : (tempX);
        int32_t zHairy = (ZCoords + hairyFactor < tempZ) ? (tempZ - 1) : (tempZ);
		// Work out the hashed gradient indices of the three simplex corners
        uint8_t ii = (uint32_t) xHairy & 0xffu;
        uint8_t jj = (uint32_t) zHairy & 0xffu;
		__prefetch_local_l1(permutations + (jj>>1));
		__prefetch_local_l1(permutations+((jj+1)>>1));
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













#endif






























#define getTempAtPos(outputDouble, xInt, zInt) outputDouble = 0;\
	generateSimplexNoise(&outputDouble, (double)xInt, (double)zInt, 1, 1, 0.02500000037252903, 0.02500000037252903, 0.25, temperature, 4);

#define getPrecipAtPos(outputDouble, xInt, zInt) outputDouble = 0;\
	generateSimplexNoise(&outputDouble, (double)xInt, (double)zInt, 1, 1, 0.25, 0.25, 0.58823529411764708, precipitation, 2);

#define getHumidityAtPos(outputDouble, xInt, zInt) outputDouble = 0;\
	generateSimplexNoise(&outputDouble, (double)xInt, (double)zInt, 1, 1, 0.05000000074505806, 0.05000000074505806, 0.33333333333333331, humidity, 4);


__device__ static inline double getTempFromTempAndPrecip(double temp, double precip) {
	double preci = precip * 1.1000000000000001 + 0.5;
	temp = (temp * 0.14999999999999999 + 0.69999999999999996) * (1.0 - 0.01) + preci * 0.01;
	
	temp = 1.0 - (1.0 - temp) * (1.0 - temp);
	if (temp < 0.0) {
		temp = 0.0;
	}
	if (temp > 1.0) {
		temp = 1.0;
	}
	return temp;
}
























#define BLOCK_SIZE 128


//WE should be able to check and immediatly remove any tundra biomes that appear as they are only if the temp is <0.1
__global__ __launch_bounds__(BLOCK_SIZE,4) static void checkSeed(uint64_t worldSeedOffset) {
	uint64_t worldSeed = blockIdx.x * blockDim.x + threadIdx.x + (worldSeedOffset << (17+8));
	
	
	register Random biomeSeed = get_random(worldSeed * 0x84a59ULL);
	double precipAtPos = getSimplexNoise(0.0, 0.0, 0.25, 0.25, 0.58823529411764708, 2, &biomeSeed);//0.0, 0.0 are the block coordinates
	
	
	
	biomeSeed = get_random(worldSeed  * 9871ULL);
	double tempAtPos = getSimplexNoise(0.0, 0.0, 0.02500000037252903, 0.02500000037252903, 0.25, 4, &biomeSeed);
	
	if (getTempFromTempAndPrecip(tempAtPos, precipAtPos) < 0.97)
		return;
	if (getTempFromTempAndPrecip(tempAtPos, precipAtPos) >0.9999999999999999)
		printf("eeeeeeeeeeee\n");
	
	//When checking plains biome, note that the temperature must be above or equal to 0.97
	return;
}
































#include <windows.h>
int main() {
	uint64_t count = 0;
	printf("doing\n");
	
	cudaSetDevice(0);
	
	//testMem<<<1,1>>>(0);
	//cudaDeviceSynchronize();
	//return 0;
	for (uint64_t seed =0; seed<5;seed++) {
		SYSTEMTIME time;
		GetSystemTime(&time);
		LONG start = (time.wSecond * 1000) + time.wMilliseconds;
		
		checkSeed<<<1ULL<<14,BLOCK_SIZE>>>(seed);
   
		cudaDeviceSynchronize();
		
		
		GetSystemTime(&time);
		LONG end = (time.wSecond * 1000) + time.wMilliseconds;
		printf("Time elapsed %dms, speed: %.2fm/s\n", (int)(end - start),((double)((1ULL<<14)*BLOCK_SIZE))/((double)(end - start))/1000.0);		
	}
	printf("%I64u\n", count);
}
















