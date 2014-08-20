#include "CudaAllPairsStaticResolver.cuh"

#include "constants.cuh"
#include "cuda_kernel.cuh"
#include <cuda.h>
#include <cmath>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <device_launch_parameters.h>



//	Modified bessel functions I0,I1,K0,K1
__device__ float mbessi0(float x) {
   float ax,ans;
   float y;

   if ((ax=fabs(x)) < 3.75f) {
      y=x/3.75f,y=y*y;
      ans=1.0f+y*(3.5156229f+y*(3.0899424f+y*(1.2067492f
         +y*(0.2659732f+y*(0.360768e-1f+y*0.45813e-2f)))));
   } else {
      y=3.75f/ax;
      ans=(exp(ax)/sqrt(ax))*(0.39894228f+y*(0.1328592e-1f
         +y*(0.225319e-2f+y*(-0.157565e-2f+y*(0.916281e-2f
         +y*(-0.2057706e-1f+y*(0.2635537e-1f+y*(-0.1647633e-1f
         +y*0.392377e-2f))))))));
   }
   return ans;
}

__device__ float mbessi1(float x) {
   float ax,ans;
   float y;


   if ((ax=fabs(x)) < 3.75) {
      y=x/3.75,y=y*y;
      ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
         +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
   } else {
      y=3.75/ax;
      ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
         -y*0.420059e-2));
      ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
         +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
      ans *= (exp(ax)/sqrt(ax));
   }
   return x < 0.0 ? -ans : ans;
}

__device__ float mbessk0(float x) {
   float y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(-log(x/2.0)*mbessi0(x))+(-0.57721566+y*(0.42278420
         +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
         +y*(0.10750e-3+y*0.74e-5))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
         +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
         +y*(-0.251540e-2+y*0.53208e-3))))));
   }
   return ans;
}

__device__ float mbessk1(float x) {
   float y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(log(x/2.0)*mbessi1(x))+(1.0/x)*(1.0+y*(0.15443144
         +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
         +y*(-0.110404e-2+y*(-0.4686e-4)))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619
         +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
         +y*(0.325614e-2+y*(-0.68245e-3)))))));
   }
   return ans;
}


//	Get mass of dark matter contained in radius r
//	according to Hernquist density profile
__device__ float dmMassAtRadius(float r, 
								float Mdm,	//	Total dark matter mass in galaxy
								float a)	//	Scale radius for Hernquist density profile
{
	return (Mdm * r * r) / powf(r + a, 2);
}

//	Get mass of stars container in radius r
//	according to density profile
__device__ float stellarMassAtRadius(float r,
									float Ms,	//	Total stellar mass in galaxy
									float Rs)	//	Scale radius for density profile
{
	return (Ms * (Rs*Rs - (Rs*r + Rs*Rs)*exp(-r/Rs))) / (Rs*Rs);
}


__global__
void cudaGenBodies(	float *d_pos,		//	Pointer to array in device memory containing positional information
					float *d_vel,		//	Pointer to array in device memory containing velocity information
					float *d_mas,		//	Pointer to array in device memory containing mass information
					float *d_rands,		//	Pointer to array in device memory containing random numbers used to generate initial conditions
					int NUM_PARTICLES,	//	Total number of particles in simulation
					float Ms,			//	Total mass of bodies in simulation
					float Rs,			//	Scale radius for stellar density profile
					float Mdm,			//	Total mass of dark matter in simulation
					float Rdm)			//	Scale radius for dark matter density profile
{

	int threadId = threadIdx.x;
	int blockId = blockIdx.x;

	int globalId = blockId * blockDim.x + threadId;

	if(globalId < NUM_PARTICLES){
		int baseIndex = globalId * 3;

		float x = d_rands[baseIndex];
		float y = d_rands[baseIndex+1];
		float z = d_rands[baseIndex+2];

		
		//	Set position

		float rx = -Rs * log(1.0f - x);

		float Sz = -(1.0f/2.0f) * (0.1f * Rs) * log(-((z-1)/z));		
		float Sx = sqrt(rx*rx) * cos(2.0f * 3.1416f * y);
		float Sy = sqrt(rx*rx) * sin(2.0f * 3.1416f * y);

		d_pos[baseIndex] = Sx;
		d_pos[baseIndex+1] = Sy;
		d_pos[baseIndex+2] = Sz;

		
		//	Set velocity

		float realRad = sqrt(Sx * Sx + Sy * Sy + Sz * Sz);
		float t = realRad / (2.0f * Rs);
		float absVel = sqrt( (G * dmMassAtRadius(realRad, Mdm, Rdm)) / realRad + ((2.0f * G * Ms) / Rs) * t * t * (mbessi0(t)*mbessk0(t) - mbessi1(t)*mbessk1(t)) );
		
		float3 velUnitVector = make_float3( - Sy / sqrt(Sx*Sx+Sy*Sy) , Sx / sqrt(Sx*Sx+Sy*Sy), 0);

		float3 velVector = make_float3(velUnitVector.x * absVel, velUnitVector.y * absVel, velUnitVector.z * absVel);

		d_vel[baseIndex] = velVector.x;
		d_vel[baseIndex+1] = velVector.y;
		d_vel[baseIndex+2] = velVector.z;

		
		//	Set mass
		//	TO-DO
	}	
}


__global__ void cudaMoveBodiesByDT_staticPotential(float *d_pos, 
												   float *d_vel, 
												   float dT, 
												   int NUM_PARTICLES, 
												   float Ms, 
												   float Rs, 
												   float Mdm, 
												   float Rdm)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;

	int globalId = blockId * blockDim.x + threadId;


	if(globalId < NUM_PARTICLES){
		float3 current_pos = make_float3(d_pos[globalId * 3], d_pos[globalId * 3 + 1], d_pos[globalId * 3 + 2]);

		current_pos.x += d_vel[globalId * 3] * velConvFactor * dT;
		current_pos.y += d_vel[globalId * 3 + 1] * velConvFactor * dT;
		current_pos.z += d_vel[globalId * 3 + 2] * velConvFactor * dT;

		d_pos[globalId * 3] = current_pos.x;
		d_pos[globalId * 3 + 1] = current_pos.y;
		d_pos[globalId * 3 + 2] = current_pos.z;

		float r = sqrt(current_pos.x * current_pos.x + current_pos.y * current_pos.y + current_pos.z * current_pos.z);
		float totalRelevantMass = 0;
		totalRelevantMass = dmMassAtRadius(r, Mdm, Rdm) + stellarMassAtRadius(r, Ms, Rs);

		float accel = -((G * totalRelevantMass) / pow(r, 2)) * (1 / kmPerPc);
		float3 accelVector = make_float3(current_pos.x / r * accel, current_pos.y / r * accel, current_pos.z / r * accel);
		
		d_vel[globalId * 3] += accelVector.x * (dT * secPerMYr);
		d_vel[globalId * 3 + 1] += accelVector.y * (dT * secPerMYr);
		d_vel[globalId * 3 + 2] += accelVector.z * (dT * secPerMYr);
	}
}


CudaAllPairsStaticResolver::CudaAllPairsStaticResolver(void)
{
	NUM_PARTICLES = -1;

	Mtot = -1;
	Msf = -1;

	MPart = -1;

	Ms = -1;
	Mdm = -1;

	Rs = -1;
	Rdm = -1;

	dT = -1;

	threadsPerBlock = -1;
}


void CudaAllPairsStaticResolver::loadSimConfig(){

	NUM_PARTICLES = 256 * 4300;

	Mtot = 96.9e10f;
	Msf = 0.14f;

	MPart = (Mtot * Msf) / NUM_PARTICLES;

	Ms = Mtot * Msf;
	Mdm = Mtot - Ms;

	Rs = 3160.0f;
	Rdm = Rs * 2;

	dT = 0.1f;

	threadsPerBlock = 256;
}

void CudaAllPairsStaticResolver::setPosBufferID(GLuint vboID){
	posVboID = vboID;
}

void CudaAllPairsStaticResolver::initialize(){

	//	Map the OpenGL VBO containing particle positions to a cuda pointer
	cudaGLRegisterBufferObject(posVboID);
	cudaGLMapBufferObject( (void **)&d_positions, posVboID);

	//	Allocate memory for velocity and mass data
	cudaMalloc(&d_velocities, 3 * NUM_PARTICLES * sizeof(float));
	cudaMalloc(&d_masses, NUM_PARTICLES * sizeof(float));


	int blocks = NUM_PARTICLES / threadsPerBlock + (NUM_PARTICLES % threadsPerBlock == 0 ? 0:1);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	float *d_randoms;
	cudaMalloc(&d_randoms, sizeof(float) * 3 * NUM_PARTICLES);

	curandGenerateUniform(gen, d_randoms, NUM_PARTICLES * 3);

	cudaGenBodies<<<blocks, threadsPerBlock>>>(d_positions, d_velocities, d_masses, d_randoms, NUM_PARTICLES, Ms, Rs, Mdm, Rdm);

	cudaFree(d_randoms);
	curandDestroyGenerator(gen);

	cudaGLUnmapBufferObject(posVboID);
}

void CudaAllPairsStaticResolver::advanceTimeStep() {
	cudaGLRegisterBufferObject(posVboID);
	cudaGLMapBufferObject( (void **)&d_positions, posVboID);

	int blocks = NUM_PARTICLES / threadsPerBlock + (NUM_PARTICLES % threadsPerBlock == 0 ? 0:1);

	cudaMoveBodiesByDT_staticPotential<<<blocks, threadsPerBlock>>>(d_positions, d_velocities, dT, NUM_PARTICLES, Ms, Rs, Mdm, Rdm);

	cudaGLUnmapBufferObject(posVboID);
}


CudaAllPairsStaticResolver::~CudaAllPairsStaticResolver(void){
	cudaFree(d_positions);
	cudaFree(d_velocities);
	cudaFree(d_masses);
}
