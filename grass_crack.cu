//"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc.exe"  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x64" -o crack crack.cu -O3 -m=64 -arch=compute_61 -code=sm_61 -Xptxas -allow-expensive-optimizations=true -Xptxas -v
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <inttypes.h>

#include <cuda.h>

#ifdef BOINC
  #include "boinc_api.h"
#if defined _WIN32 || defined _WIN64
  #include "boinc_win.h"
#endif
#endif


// ===== LCG IMPLEMENTATION ===== //

namespace java_lcg { //region Java LCG
    #define Random uint64_t
    #define RANDOM_MULTIPLIER 0x5DEECE66DULL
    #define RANDOM_ADDEND 0xBULL
    #define RANDOM_MASK ((1ULL << 48u) - 1)
    #define get_random(seed) ((Random)((seed ^ RANDOM_MULTIPLIER) & RANDOM_MASK))


    __host__ __device__ __forceinline__ static int32_t random_next(Random *random, int bits) {
        *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
        return (int32_t) (*random >> (48u - bits));
    }
    __device__ __forceinline__ static int32_t random_next_int(Random *random, const uint16_t bound) {
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
    /*
    __device__ __forceinline__ static int32_t random_next_int(Random *random, const uint16_t bound) {
        int32_t r = random_next(random, 31);
        if (__popc(bound) == 1) {
            return (int32_t) ((bound * (uint64_t) r) >> 31u);
        } else {
            const uint16_t m = bound - 1u;
            for (int32_t u = r;
                 u - (r = u % bound) + m < 0;
                 u = random_next(random, 31));
        }
        return r;
    }*/
    __host__ __device__ __forceinline__ static double next_double(Random *random) {
        return (double) ((((uint64_t) ((uint32_t) random_next(random, 26)) << 27u)) + random_next(random, 27)) / (double)(1ULL << 53);
    }
    __host__ __device__ __forceinline__ static uint64_t random_next_long (Random *random) {
        return (((uint64_t)random_next(random, 32)) << 32u) + (int32_t)random_next(random, 32);
    }
    __host__ __device__ __forceinline__ static void advance2(Random *random) {
        *random = (*random * 0xBB20B4600A69LLU + 0x40942DE6BALLU) & RANDOM_MASK;
    }

}
using namespace java_lcg;


namespace device_intrinsics { //region DEVICE INTRINSICS
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
}
using namespace device_intrinsics;



#define BLOCK_SIZE (128)
//#define BLOCK_SIZE (64)
#define WORK_SIZE_BITS 15
#define SEEDS_PER_CALL ((1ULL << (WORK_SIZE_BITS)) * BLOCK_SIZE)




//The generation of the simplex layers and noise
namespace simplex { //region Simplex layer gen
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
    
    struct SimplexOctave {
        double xo;
        double yo;
        uint8_t permutations[256];
    };

    __shared__ uint8_t permutations[256][BLOCK_SIZE];


    #define getValue(array, index) array[index][threadIdx.x]
    #define setValue(array, index, value) array[index][threadIdx.x] = value



    /* simplex noise result is in buffer */
    __device__ static inline double getSimplexNoise(const double chunkX, const double chunkZ, double offsetX, double offsetZ, const double ampFactor, const uint8_t nbOctaves, Random *random, SimplexOctave resultArray[]) {
        offsetX /= 1.5;
        offsetZ /= 1.5;
        double res = 0.0;
        double octaveDiminution = 1.0;
        double octaveAmplification = 1.0;
        for (int j = 0; j < nbOctaves; ++j) {
            __prefetch_local_l2(&resultArray[j]);
            double xo = next_double(random) * 256.0;
            double yo = next_double(random) * 256.0;
            
            
            advance2(random);
            #pragma unroll
            for(int w = 0; w<256; w++) {
                setValue(permutations, w, w);
            }
            for(int index = 0; index<256; index++) {
                uint32_t randomIndex = random_next_int(random, 256ull - index) + index;
                //if (randomIndex != index) {
                    // swap
                    uint8_t v1 = getValue(permutations,index);
                    uint8_t v2 = getValue(permutations,randomIndex);
                    setValue(permutations,index, v2);
                    setValue(permutations, randomIndex, v1);
                //}
            }
            double XCoords = (double) chunkX * offsetX * octaveAmplification + xo;
            double ZCoords = (double) chunkZ * offsetZ * octaveAmplification + yo;
            // Skew the input space to determine which simplex cell we're in
            double hairyFactor = (XCoords + ZCoords) * F2;
            int32_t tempX = static_cast<int32_t>(XCoords + hairyFactor);
            int32_t tempZ = static_cast<int32_t>(ZCoords + hairyFactor);
            int32_t xHairy = (XCoords + hairyFactor < tempX) ? (tempX - 1) : (tempX);
            int32_t zHairy = (ZCoords + hairyFactor < tempZ) ? (tempZ - 1) : (tempZ);
            // Work out the hashed gradient indices of the three simplex corners
            uint32_t ii = (uint32_t) xHairy & 0xffu;
            uint32_t jj = (uint32_t) zHairy & 0xffu;
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

            
            uint8_t gi0 = getValue(permutations,(uint32_t) (ii + getValue(permutations,jj)) & 0xffu) % 12u;
            uint8_t gi1 = getValue(permutations,(uint32_t)(ii + offsetSecondCornerX + getValue(permutations,(uint32_t) (jj + offsetSecondCornerZ) & 0xffu))& 0xffu) % 12u;
            uint8_t gi2 = getValue(permutations,(uint32_t)(ii + 1 + getValue(permutations,(uint32_t)(jj + 1)& 0xffu))& 0xffu) % 12u;

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
            
            resultArray[j].xo = xo;
            resultArray[j].yo = yo;
            #pragma unroll
            for(int c = 0; c<256;c++) {
                __prefetch_local_l1(&(resultArray[j].permutations[c+1]));
                resultArray[j].permutations[c] = getValue(permutations,c);
            }
        }
        return res;

    }



    __device__ static inline double getSimplexNoiseFromOctave(const double chunkX, const double chunkZ, double offsetX, double offsetZ, const double ampFactor, const uint8_t nbOctaves, const SimplexOctave resultArray[]) {
        __prefetch_local_l1(&resultArray[0]);//Double check
        offsetX /= 1.5;
        offsetZ /= 1.5;
        double res = 0.0;
        double octaveDiminution = 1.0;
        double octaveAmplification = 1.0;
        for (uint8_t j = 0; j < nbOctaves; ++j) {
            __prefetch_local_l2(&resultArray[j+1]);
            double xo = resultArray[j].xo;
            double yo = resultArray[j].yo;
            
            double XCoords = (double) chunkX * offsetX * octaveAmplification + xo;
            double ZCoords = (double) chunkZ * offsetZ * octaveAmplification + yo;
            // Skew the input space to determine which simplex cell we're in
            double hairyFactor = (XCoords + ZCoords) * F2;
            int32_t tempX = static_cast<int32_t>(XCoords + hairyFactor);
            int32_t tempZ = static_cast<int32_t>(ZCoords + hairyFactor);
            int32_t xHairy = (XCoords + hairyFactor < tempX) ? (tempX - 1) : (tempX);
            int32_t zHairy = (ZCoords + hairyFactor < tempZ) ? (tempZ - 1) : (tempZ);
            // Work out the hashed gradient indices of the three simplex corners
            uint8_t ii = (uint32_t) xHairy & 0xffu;
            uint8_t jj = (uint32_t) zHairy & 0xffu;
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

            
            uint8_t gi0 = resultArray[j].permutations[(uint16_t) (ii + resultArray[j].permutations[jj]) & 0xffu] % 12u;
            uint8_t gi1 = resultArray[j].permutations[(uint16_t)(ii + offsetSecondCornerX + resultArray[j].permutations[(uint16_t) (jj + offsetSecondCornerZ) & 0xffu])& 0xffu] % 12u;
            uint8_t gi2 = resultArray[j].permutations[(uint16_t)(ii + 1 + resultArray[j].permutations[(uint16_t)(jj + 1)& 0xffu])& 0xffu] % 12u;

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
}
using namespace simplex;


namespace more_simplex {
    #define getSimplexInital(x,y,a1,a2,a3,layer_count,seed,out_array) getSimplexNoise(x,y,a1,a2,a3,layer_count,seed,out_array)
    #define getSimplex(x,y,a1,a2,a3,layer_count,data_array) getSimplexNoiseFromOctave(x,y,a1,a2,a3,layer_count,data_array)

    #define getSimplexHumidtyInital(x,y,seed,out_array) getSimplexInital(x,y,0.05000000074505806, 0.05000000074505806, 0.33333333333333331, 4, seed, out_array)
    #define getSimplexHumidty(x,y,data_array) getSimplex(x,y,0.05000000074505806, 0.05000000074505806, 0.33333333333333331, 4, data_array)
	
	
	__constant__ uint8_t const biomeLookup[] = {11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1};
	__device__ static inline uint8_t getBiome(int x, int z, SimplexOctave precipOctaves[], SimplexOctave tempOctaves[], SimplexOctave humidOctaves[]) {
		double precipAtPos =  getSimplex((double)x, (double)z, 0.25, 0.25, 0.58823529411764708, 2, precipOctaves);
		double tempAtPos = getSimplex((double)x, (double)z, 0.02500000037252903, 0.02500000037252903, 0.25, 4, tempOctaves);
		double humidityAtPos = getSimplex((double)x, (double)z, 0.05000000074505806, 0.05000000074505806, 0.33333333333333331, 4, humidOctaves);
		int32_t index = ConvertToIndex(getTempFromTempAndPrecip(tempAtPos, precipAtPos)) + ConvertToIndex(getHumidFromHumidAndPrecip(humidityAtPos,precipAtPos)) * 64;
		return __ldg(&biomeLookup[index]);
	}
}
using namespace more_simplex;

#define ABS_PRECIP (3*.55)
#define MIN_PRECIP (-ABS_PRECIP * 1.1 + 0.5)
#define MAX_PRECIP (ABS_PRECIP * 1.1 + 0.5)
#define D1 0.002
#define D2 (1 - D1)
#define decodeMinHumid(minHumid) ((((minHumid) - MAX_PRECIP * D1) / D2 - 0.5) / 0.15)
#define decodeMaxHumid(maxHumid) ((((maxHumid) - MIN_PRECIP * D1) / D2 - 0.5) / 0.15)

#define GRASS1_X 64
#define GRASS1_Z (-53)
#define GRASS1_MIN_HUMID decodeMinHumid(0.2723577235772357)
#define GRASS1_MAX_HUMID decodeMaxHumid(0.325)

#define GRASS2_X 59
#define GRASS2_Z (-19)
#define GRASS2_MIN_HUMID decodeMinHumid(0.44313725490196076)
#define GRASS2_MAX_HUMID decodeMaxHumid(0.5081967213114754)

#define GRASS3_X 83
#define GRASS3_Z (-40)
#define GRASS3_MIN_HUMID decodeMinHumid(0.4117647058823529)
#define GRASS3_MAX_HUMID decodeMaxHumid(0.4833333333333334)



#define PLAINS_BIOME_PLAYER_X 61
#define PLAINS_BIOME_PLAYER_Z -68

#define PLAINS_BIOME_X 48
#define PLAINS_BIOME_Z -72

#define DESERT_BIOME_X 47
#define DESERT_BIOME_Z -72

#define PLAINS_FOREST_BIOME_2_X 33
#define PLAINS_FOREST_BIOME_2_Z -82

#define DESERT_BIOME_2_X 33
#define DESERT_BIOME_2_Z -81


// //RANDOMLY CHOOSEN, GET ACTUALL DESERT COORDS


//Test humidity
__global__ __launch_bounds__(BLOCK_SIZE,4) static void checkSeedBiomesHumidity(uint64_t worldSeedOffset, uint32_t* count, uint64_t* seeds) {
    int64_t seed = blockIdx.x * blockDim.x + threadIdx.x + worldSeedOffset;
    
        
    register Random biomeSeed = get_random(seed  * 39811LL);
    SimplexOctave humidOct[4];
    double humidAtPos = getSimplexHumidtyInital((double)GRASS3_X, (double)GRASS3_Z, &biomeSeed, humidOct);
    //Plains biome humidity check
    if (!(GRASS3_MIN_HUMID<humidAtPos&&humidAtPos<GRASS3_MAX_HUMID)) {
        return;
    }

#define testHumidity(x, z, min, max) humidAtPos = getSimplexHumidty((double)x, (double)z, humidOct);\
if (!(min < humidAtPos && humidAtPos < max)) return;

    testHumidity(GRASS2_X, GRASS2_Z, GRASS2_MIN_HUMID, GRASS2_MAX_HUMID)
    testHumidity(GRASS1_X, GRASS1_Z, GRASS1_MIN_HUMID, GRASS1_MAX_HUMID)
    
    seeds[atomicAdd(count, 1)] = seed;
}

//Test temperature and other points

__global__ __launch_bounds__(BLOCK_SIZE,2) static void part2ElectricBooglo(uint64_t worldSeedOffset, uint32_t count, uint64_t* seeds) {
    if (blockIdx.x * blockDim.x + threadIdx.x >= count)
        return;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t seed = seeds[index];
    
	//REGION: check if the player is in a plains biome
	
	SimplexOctave tempOct[4];
	SimplexOctave precipOct[2];
	SimplexOctave humidOct[4];
    {
		register Random biomeSeed = get_random(seed  * 9871LL);
		double tempAtPos = getSimplexNoise((double)PLAINS_BIOME_PLAYER_X, (double)PLAINS_BIOME_PLAYER_Z, 0.02500000037252903, 0.02500000037252903, 0.25, 4, &biomeSeed, tempOct);
		if (!(1.06<tempAtPos&&tempAtPos<3.006)) {
			seeds[index] = 0;
			return;
		}
		biomeSeed = get_random(seed  * 0x84a59LL);
		double precipAtPos = getSimplexNoise((double)PLAINS_BIOME_PLAYER_X, (double)PLAINS_BIOME_PLAYER_Z, 0.25, 0.25, 0.58823529411764708, 2, &biomeSeed, precipOct);
		//If its not a plains biome
		if (ConvertToIndex(getTempFromTempAndPrecip(tempAtPos, precipAtPos))<62) {
			seeds[index] = 0;
			return;
		}
		
		
		biomeSeed = get_random(seed  * 39811LL);
		double humidAtPos = getSimplexHumidtyInital((double)PLAINS_BIOME_PLAYER_X, (double)PLAINS_BIOME_PLAYER_Z, &biomeSeed, humidOct);
		int32_t humid_index = ConvertToIndex(getHumidFromHumidAndPrecip(humidAtPos, precipAtPos));
		if (!(12 < humid_index && humid_index < 29)) {
			seeds[index] = 0;
			return;	
		}
	}
	
	
	if (getBiome(DESERT_BIOME_X, DESERT_BIOME_Z, precipOct, tempOct, humidOct)!=8) {
		seeds[index] = 0;
        return;	
	}
	
	int biome_num = getBiome(PLAINS_BIOME_X, PLAINS_BIOME_Z, precipOct, tempOct, humidOct);
	if (!(biome_num==9||biome_num==6)) {
		seeds[index] = 0;
        return;	
	}
	
	
	if (getBiome(DESERT_BIOME_X, DESERT_BIOME_Z, precipOct, tempOct, humidOct)!=8) {
		seeds[index] = 0;
        return;	
	}
	
	if (getBiome(DESERT_BIOME_2_X, DESERT_BIOME_2_Z, precipOct, tempOct, humidOct)!=8) {
		seeds[index] = 0;
        return;	
	}
	biome_num = getBiome(PLAINS_FOREST_BIOME_2_X, PLAINS_FOREST_BIOME_2_Z, precipOct, tempOct, humidOct);
	if (!(biome_num==9||biome_num==4||biome_num==6)) {
		seeds[index] = 0;
        return;	
	}
 
}


namespace host_processing { //region Host side processing

    #ifdef BOINC
    bool setCudaBlockingSync(int device) {
        CUdevice  hcuDevice;
        CUcontext hcuContext;

        CUresult status = cuInit(0);
        if(status != CUDA_SUCCESS)
           return false;

        status = cuDeviceGet( &hcuDevice, device);
        if(status != CUDA_SUCCESS)
           return false;

        status = cuCtxCreate( &hcuContext, 0x4, hcuDevice );
        if(status != CUDA_SUCCESS)
           return false;

        return true;
    }
    #endif
    #ifndef BOINC
    #define boinc_begin_critical_section()
    #define boinc_end_critical_section()
    #define boinc_finish(status)
    #define boinc_fraction_done(fraction)
    #endif

    #define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
    inline void gpuAssert(cudaError_t code, const char *file, int line) {
      if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        boinc_finish(code);
        #ifndef BOINC
        exit(code);
        #endif
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


    uint32_t actual_count = 0;
    int host_main(int argc, char** argv) {

        #ifdef BOINC
        BOINC_OPTIONS options;

        boinc_options_defaults(options);
        options.normal_thread_priority = true;
        boinc_init_options(&options);
        #endif

        if (argc < 3) {
            fprintf(stderr, "Not enough arguments\n");
            return 2;
        }
        int start_batch = atoi(argv[1]);
        int end_batch = atoi(argv[2]);
        if (start_batch < 0 || start_batch >= end_batch || end_batch > (1ULL << 48) / SEEDS_PER_CALL) {
            fprintf(stderr, "Invalid batch bounds: %d to %d\n", start_batch, end_batch);
            return 1;
        }

        fprintf(stderr, "doing between %lld (inclusive) and %lld (exclusive)\n", start_batch * SEEDS_PER_CALL, end_batch * SEEDS_PER_CALL);

        int gpu_device = 0;

        #ifdef BOINC
        APP_INIT_DATA aid;
        boinc_get_init_data(aid);
        if (aid.gpu_device_num >= 0) {
            gpu_device = aid.gpu_device_num;
            fprintf(stderr,"boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, gpu_device);
        } else {
            fprintf(stderr,"stdalone gpuindex % \n", gpu_device);
        }

        setCudaBlockingSync(gpu_device);
        #endif
        cudaSetDevice(gpu_device);
        GPU_ASSERT(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
        //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        GPU_ASSERT(cudaPeekAtLastError());
        GPU_ASSERT(cudaDeviceSynchronize());
        
        
        uint32_t* count;
        GPU_ASSERT(cudaMallocManaged(&count, sizeof(*count)));
        GPU_ASSERT(cudaPeekAtLastError());
        
        uint64_t* seedBuffer;
        GPU_ASSERT(cudaMallocManaged(&seedBuffer, sizeof(*seedBuffer) * (SEEDS_PER_CALL>>5)));//5 is an estimate taken from the number of seeds filtered
        GPU_ASSERT(cudaPeekAtLastError());
        
        
        for (uint64_t seed = start_batch * SEEDS_PER_CALL, end_seed = end_batch * SEEDS_PER_CALL; seed < end_seed; seed+=SEEDS_PER_CALL) {
            uint64_t start = getCurrentTimeMillis();
            
            boinc_begin_critical_section();
            *count = 0;
            checkSeedBiomesHumidity<<< 1ULL << WORK_SIZE_BITS, BLOCK_SIZE>>>(seed, count, seedBuffer); // produces about 32k seeds per call
            GPU_ASSERT(cudaPeekAtLastError());
            GPU_ASSERT(cudaDeviceSynchronize());
            //Double check work size calculation
            part2ElectricBooglo<<< ceil(((double)*count)/BLOCK_SIZE), BLOCK_SIZE>>>(seed, *count, seedBuffer);
            GPU_ASSERT(cudaPeekAtLastError());
            GPU_ASSERT(cudaDeviceSynchronize());
            //uint32_t actual_count = 0;
            for(uint32_t i = 0; i<*count;i++) {
                uint64_t seed = seedBuffer[i];
                if( seed != 0) {
                    actual_count ++;
					fprintf(stderr, "SEED FOUND: %lld\n",seed);
                }               
            }
            boinc_end_critical_section();
            
            uint64_t end = getCurrentTimeMillis();
            double fraction_done = ((double)(seed-(start_batch * SEEDS_PER_CALL)))/((end_batch * SEEDS_PER_CALL)-(start_batch * SEEDS_PER_CALL));
            printf("Time elapsed %dms, speed: %.2fm/s, seed count 1: %i, seed count 2: %i, percent done: %f\n", (int)(end - start),((double)((1ULL<<WORK_SIZE_BITS)*(BLOCK_SIZE)))/((double)(end - start))/1000.0,*count, actual_count, fraction_done*100);      
            if ((seed / SEEDS_PER_CALL) % 30) { // about every 15 seconds
                boinc_fraction_done(fraction_done);
            }
        }
        fprintf(stderr, "Finished work unit\n");
        boinc_finish(0);
        return 0;
    }
}
using namespace host_processing;
int main(int argc, char** argv) { return host_main(argc, argv); }
