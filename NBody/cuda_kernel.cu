//#include "cuda_kernel.cuh"
//#include <cmath>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include <cuda_gl_interop.h>
//#include <curand.h>
//#include <device_launch_parameters.h>
//
//
//
//__device__ const float kmPerPc = 3.0857e13;		//	Kilometers per Parsec
//__device__ const float G = 4.302e-3;			//	Gravitational constant in ( pc / SM ) * (km/s)^2
//__device__ const float velConvFactor = 1.0226;	//	Conversion factor from km/s to pc/Myr
//__device__ const float secPerMYr = 3.15569e13;	//	Number of seconds per Myr
//
////	Modified bessel functions I0,I1,K0,K1
//__device__ float mbessi0(float x) {
//   float ax,ans;
//   float y;
//
//   if ((ax=fabs(x)) < 3.75f) {
//      y=x/3.75f,y=y*y;
//      ans=1.0f+y*(3.5156229f+y*(3.0899424f+y*(1.2067492f
//         +y*(0.2659732f+y*(0.360768e-1f+y*0.45813e-2f)))));
//   } else {
//      y=3.75f/ax;
//      ans=(exp(ax)/sqrt(ax))*(0.39894228f+y*(0.1328592e-1f
//         +y*(0.225319e-2f+y*(-0.157565e-2f+y*(0.916281e-2f
//         +y*(-0.2057706e-1f+y*(0.2635537e-1f+y*(-0.1647633e-1f
//         +y*0.392377e-2f))))))));
//   }
//   return ans;
//}
//
//__device__ float mbessi1(float x) {
//   float ax,ans;
//   float y;
//
//
//   if ((ax=fabs(x)) < 3.75) {
//      y=x/3.75,y=y*y;
//      ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
//         +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
//   } else {
//      y=3.75/ax;
//      ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
//         -y*0.420059e-2));
//      ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
//         +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
//      ans *= (exp(ax)/sqrt(ax));
//   }
//   return x < 0.0 ? -ans : ans;
//}
//
//__device__ float mbessk0(float x) {
//   float y,ans;
//
//   if (x <= 2.0) {
//      y=x*x/4.0;
//      ans=(-log(x/2.0)*mbessi0(x))+(-0.57721566+y*(0.42278420
//         +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
//         +y*(0.10750e-3+y*0.74e-5))))));
//   } else {
//      y=2.0/x;
//      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
//         +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
//         +y*(-0.251540e-2+y*0.53208e-3))))));
//   }
//   return ans;
//}
//
//__device__ float mbessk1(float x) {
//   float y,ans;
//
//   if (x <= 2.0) {
//      y=x*x/4.0;
//      ans=(log(x/2.0)*mbessi1(x))+(1.0/x)*(1.0+y*(0.15443144
//         +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
//         +y*(-0.110404e-2+y*(-0.4686e-4)))))));
//   } else {
//      y=2.0/x;
//      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619
//         +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
//         +y*(0.325614e-2+y*(-0.68245e-3)))))));
//   }
//   return ans;
//}
//
////	Get mass of dark matter contained in radius r
////	according to Hernquist density profile
//__device__ float dmMassAtRadius(float r, 
//								float Mdm,	//	Total dark matter mass in galaxy
//								float a)	//	Scale radius for Hernquist density profile
//{
//	return (Mdm * r * r) / powf(r + a, 2);
//}
//
////	Get mass of stars container in radius r
////	according to density profile
//__device__ float galaxyMassAtRadius(float r,
//									float Ms,	//	Total stellar mass in galaxy
//									float Rs)	//	Scale radius for density profile
//{
//	return (Ms * (Rs*Rs - (Rs*r + Rs*Rs)*exp(-r/Rs))) / (Rs*Rs);
//}
//
//__global__ void cudaGenBodies(float *d_pos, float *d_vel, float *d_rands, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm){
//
//	int threadId = threadIdx.x;
//	int blockId = blockIdx.x;
//
//	int globalId = blockId * blockDim.x + threadId;
//
//	if(globalId < NUM_PARTICLES){
//		int baseIndex = globalId * 3;
//
//		float x = d_rands[baseIndex];
//		float y = d_rands[baseIndex+1];
//		float z = d_rands[baseIndex+2];
//
//		
//		//	Set position
//
//		float rx = -Rs * log(1.0f - x);
//
//		float Sz = -(1.0f/2.0f) * (0.1f * Rs) * log(-((z-1)/z));		
//		float Sx = sqrt(rx*rx) * cos(2.0f * 3.1416f * y);
//		float Sy = sqrt(rx*rx) * sin(2.0f * 3.1416f * y);
//
//		d_pos[baseIndex] = Sx;
//		d_pos[baseIndex+1] = Sy;
//		d_pos[baseIndex+2] = Sz;
//
//		
//		//	Set velocity
//
//		float realRad = sqrt(Sx * Sx + Sy * Sy + Sz * Sz);
//		float t = realRad / (2.0f * Rs);
//		float absVel = sqrt( (G * dmMassAtRadius(realRad, Mdm, Rdm)) / realRad + ((2.0f * G * Ms) / Rs) * t * t * (mbessi0(t)*mbessk0(t) - mbessi1(t)*mbessk1(t)) );
//		
//		float3 velUnitVector = make_float3( - Sy / sqrt(Sx*Sx+Sy*Sy) , Sx / sqrt(Sx*Sx+Sy*Sy), 0);
//
//		float3 velVector = make_float3(velUnitVector.x * absVel, velUnitVector.y * absVel, velUnitVector.z * absVel);
//
//		d_vel[baseIndex] = velVector.x;
//		d_vel[baseIndex+1] = velVector.y;
//		d_vel[baseIndex+2] = velVector.z;
//	}	
//}
//
//__global__ void cudaMoveBodiesByDT_staticPotential(float *d_pos, float *d_vel, float dT, float bodyMass, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm){
//	int threadId = threadIdx.x;
//	int blockId = blockIdx.x;
//
//	int globalId = blockId * blockDim.x + threadId;
//
//
//	if(globalId < NUM_PARTICLES){
//		float3 current_pos = make_float3(d_pos[globalId * 3], d_pos[globalId * 3 + 1], d_pos[globalId * 3 + 2]);
//
//		current_pos.x += d_vel[globalId * 3] * velConvFactor * dT;
//		current_pos.y += d_vel[globalId * 3 + 1] * velConvFactor * dT;
//		current_pos.z += d_vel[globalId * 3 + 2] * velConvFactor * dT;
//
//		d_pos[globalId * 3] = current_pos.x;
//		d_pos[globalId * 3 + 1] = current_pos.y;
//		d_pos[globalId * 3 + 2] = current_pos.z;
//
//		float r = sqrt(current_pos.x * current_pos.x + current_pos.y * current_pos.y + current_pos.z * current_pos.z);
//		float totalRelevantMass = 0;
//		totalRelevantMass = dmMassAtRadius(r, Mdm, Rdm);// + galaxyMassAtRadius(r, Ms, Rs);
//
//		float accel = -((G * totalRelevantMass) / pow(r, 2)) * (1 / kmPerPc);
//		float3 accelVector = make_float3(current_pos.x / r * accel, current_pos.y / r * accel, current_pos.z / r * accel);
//		
//		d_vel[globalId * 3] += accelVector.x * (dT * secPerMYr);
//		d_vel[globalId * 3 + 1] += accelVector.y * (dT * secPerMYr);
//		d_vel[globalId * 3 + 2] += accelVector.z * (dT * secPerMYr);
//	}
//}
//
//__global__ void cudaMoveBodiesByDT_NBody(float *d_pos, float *d_vel, float dT, float bodyMass, float Mdm, float Rdm, int NUM_PARTICLES){
//	int threadId = threadIdx.x;
//	int blockId = blockIdx.x;
//
//	int globalId = blockId * blockDim.x + threadId;
//
//	extern __shared__ float shmem[];
//
//	if(globalId < NUM_PARTICLES){
//
//		float3 currentParticlePos = make_float3(d_pos[globalId * 3], d_pos[globalId * 3 + 1], d_pos[globalId * 3 + 2]);
//
//		currentParticlePos.x += d_vel[globalId * 3] * velConvFactor * dT;
//		currentParticlePos.y += d_vel[globalId * 3 + 1] * velConvFactor * dT;
//		currentParticlePos.z += d_vel[globalId * 3 + 2] * velConvFactor * dT;
//
//		float3 totalAcceleration = make_float3(0, 0, 0);	//	in km/s
//
//		//	Stellar gravitational influences
//		
//		for(int stride = 0; stride < NUM_PARTICLES - blockDim.x; stride += blockDim.x){
//			__syncthreads();
//
//			shmem[threadId * 3] = d_pos[(stride + threadId) * 3];
//			shmem[threadId * 3 + 1] = d_pos[(stride + threadId) * 3 + 1];
//			shmem[threadId * 3 + 2] = d_pos[(stride + threadId) * 3 + 2];
//
//			__syncthreads();
//			for(int i = 0; i < blockDim.x; i++){
//				if(globalId != (stride + i)){
//					float3 destParticlePos = make_float3(shmem[i * 3], shmem[i * 3 + 1], shmem[i * 3 + 2]);
//
//					float3 rVector = make_float3(currentParticlePos.x - destParticlePos.x, currentParticlePos.y - destParticlePos.y, currentParticlePos.z - destParticlePos.z);
//					float r = sqrtf(rVector.x * rVector.x + rVector.y * rVector.y + rVector.z * rVector.z);
//					float3 rUnit = make_float3(rVector.x / r, rVector.y / r, rVector.z / r);
//
//					//float acc = -((G * bodyMass) / (r*r)) * (1 / kmPerPc);
//					float a = 0.6f;
//					float acc = -((G * bodyMass * r) / ( sqrtf(powf(r*r + a * a, 3)) )) * (1 / kmPerPc);
//
//					totalAcceleration.x += acc * rUnit.x;
//					totalAcceleration.y += acc * rUnit.y;
//					totalAcceleration.z += acc * rUnit.z;
//				}
//			}
//		}
//		
//		//	---
//
//		//	Dark Matter Gravitational influence
//		
//		float3 rVector = currentParticlePos;
//		float r = sqrtf(rVector.x * rVector.x + rVector.y * rVector.y + rVector.z * rVector.z);
//		float3 rUnit = make_float3(rVector.x / r, rVector.y / r, rVector.z / r);
//
//		float relevantDMMass = dmMassAtRadius(r, Mdm, Rdm);
//
//		float accFromDM = -((G * relevantDMMass) / (r * r)) * (1 / kmPerPc);
//
//		totalAcceleration.x += accFromDM * rUnit.x;
//		totalAcceleration.y += accFromDM * rUnit.y;
//		totalAcceleration.z += accFromDM * rUnit.z;
//		
//		//	---
//
//		d_vel[globalId * 3] += totalAcceleration.x * (dT * secPerMYr);
//		d_vel[globalId * 3 + 1] += totalAcceleration.y * (dT * secPerMYr);
//		d_vel[globalId * 3 + 2] += totalAcceleration.z * (dT * secPerMYr);
//
//		d_pos[globalId * 3] = currentParticlePos.x;
//		d_pos[globalId * 3 + 1] = currentParticlePos.y;
//		d_pos[globalId * 3 + 2] = currentParticlePos.z;
//	}
//}
//
//void genBodies(GLuint posVBO, GLuint velVBO, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm){
//
//	cudaGLRegisterBufferObject(posVBO);
//	cudaGLRegisterBufferObject(velVBO);
//	float *d_pos;
//	float *d_vel;
//	cudaGLMapBufferObject( (void **)&d_pos, posVBO);
//	cudaGLMapBufferObject( (void **)&d_vel, velVBO);
//
//
//	int blockSize = 256;
//	int blocks = NUM_PARTICLES / blockSize + (NUM_PARTICLES % blockSize == 0 ? 0:1);
//
//	curandGenerator_t gen;
//	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
//
//	float *d_randoms;
//	cudaMalloc(&d_randoms, sizeof(float) * 3 * NUM_PARTICLES);
//
//	curandGenerateUniform(gen, d_randoms, NUM_PARTICLES * 3);
//
//	cudaGenBodies<<<blocks, blockSize>>>(d_pos, d_vel, d_randoms, NUM_PARTICLES, Ms, Rs, Mdm, Rdm);
//
//	cudaFree(d_randoms);
//	curandDestroyGenerator(gen);
//
//	cudaGLUnmapBufferObject(posVBO);
//	cudaGLUnmapBufferObject(velVBO);
//}
//
//void moveBodiesByDT_staticPotential(GLuint posVBO, GLuint velVBO, float dT, float bodyMass, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm){
//	cudaGLRegisterBufferObject(posVBO);
//	cudaGLRegisterBufferObject(velVBO);
//	float *d_pos;
//	float *d_vel;
//	cudaGLMapBufferObject( (void **)&d_pos, posVBO);
//	cudaGLMapBufferObject( (void **)&d_vel, velVBO);
//
//
//	int blockSize = 256;
//	int blocks = NUM_PARTICLES / blockSize + (NUM_PARTICLES % blockSize == 0 ? 0:1);
//
//	cudaMoveBodiesByDT_staticPotential<<<blocks, blockSize>>>(d_pos, d_vel, dT, bodyMass, NUM_PARTICLES, Ms, Rs, Mdm, Rdm);
//
//	cudaGLUnmapBufferObject(posVBO);
//	cudaGLUnmapBufferObject(velVBO);
//}
//
//void moveBodiesByDT_NBody(GLuint posVBO, GLuint velVBO, float dT, float bodyMass, float Mdm, float Rdm, int NUM_PARTICLES){
//	cudaGLRegisterBufferObject(posVBO);
//	cudaGLRegisterBufferObject(velVBO);
//	float *d_pos;
//	float *d_vel;
//	cudaGLMapBufferObject( (void **)&d_pos, posVBO);
//	cudaGLMapBufferObject( (void **)&d_vel, velVBO);
//
//
//	int blockSize = 256;
//	int blocks = NUM_PARTICLES / blockSize + (NUM_PARTICLES % blockSize == 0 ? 0:1);
//
//	int shmem = blockSize * 3 * sizeof(float);
//
//	cudaMoveBodiesByDT_NBody<<<blocks, blockSize, shmem>>>(d_pos, d_vel, dT, bodyMass, Mdm, Rdm, NUM_PARTICLES);
//
//
//	cudaGLUnmapBufferObject(posVBO);
//	cudaGLUnmapBufferObject(velVBO);
//}