#include <iostream>
#include "cuda_kernel.cuh"
#include <cmath>
#include <cuda.h>
#include <curand.h>
#include <device_launch_parameters.h>

__device__ const float kmPerPc = 3.0857e13;

__global__ void cudaGenBodies(float *d_pos, float *d_rands, float Rs, int SIZE){

	int threadId = threadIdx.x;
	int blockId = blockIdx.x;

	int globalId = blockId * blockDim.x + threadId;

	if(globalId < SIZE){
		int baseIndex = globalId * 3;

		float x = d_rands[baseIndex];
		float y = d_rands[baseIndex+1];
		float z = d_rands[baseIndex+2];

		float rx = -Rs * log(1.0f - x);

		float Sz = -(1.0f/2.0f) * 0.1f * Rs * log(-((z-1)/z));		
		float Sx = sqrt(rx*rx) * cos(2.0f * 3.1416f * y);
		float Sy = sqrt(rx*rx) * sin(2.0f * 3.1416f * y);

		d_pos[baseIndex] = Sx;
		d_pos[baseIndex+1] = Sy;
		d_pos[baseIndex+2] = Sz;
	}	
}

void genBodies(float *d_pos, float Rs, int SIZE){

	int blockSize = 256;
	int blocks = SIZE / blockSize + (SIZE % blockSize == 0 ? 0:1);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	float *d_randoms;
	cudaMalloc(&d_randoms, sizeof(float) * 3 * SIZE);

	curandGenerateUniform(gen, d_randoms, SIZE * 3);

	cudaGenBodies<<<blocks, blockSize>>>(d_pos, d_randoms, Rs, SIZE);

	cudaFree(d_randoms);
	curandDestroyGenerator(gen);
}