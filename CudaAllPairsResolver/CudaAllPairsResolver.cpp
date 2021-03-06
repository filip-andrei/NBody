#include "CudaAllPairsResolver.h"
#include "CudaAllPairsResolver.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>

#ifdef _DEBUG
#pragma comment(lib, "libyaml-cppmdd.lib")
#else
#pragma comment(lib, "libyaml-cppmd.lib")
#endif

using namespace std;


CudaAllPairsResolver::CudaAllPairsResolver(void)
{

}


void CudaAllPairsResolver::setPosBufferID(GLuint vboID){
	posVboID = vboID;
}

bool CudaAllPairsResolver::initialize(YAML::Node &config){
	
	try{

		//	Load sim config
		if(config["NumParticles"]){
			NUM_PARTICLES = config["NumParticles"].as<int>();
		}else{
			cout<<"Need NumParticles"<<endl;
			return false;
		}

		
		if(config["ResolverSettings"]){

			YAML::Node &resolverConfig = config["ResolverSettings"];

			if(resolverConfig["Mtot"]){
				Mtot = resolverConfig["Mtot"].as<float>();
			}else{
				cout<<"Mtot not found in config file"<<endl;
				return false;
			}

			if(resolverConfig["Msf"]){
				Msf = resolverConfig["Msf"].as<float>();
			}else{
				cout<<"Msf not found in config file"<<endl;
				return false;
			}
			
			if(resolverConfig["Rs"]){
				Rs = resolverConfig["Rs"].as<float>();
			}else{
				cout<<"Rs not found in config file"<<endl;
				return false;
			}
			
			if(resolverConfig["Rdm"]){
				Rdm = resolverConfig["Rdm"].as<float>();
			}else{
				cout<<"Rdm not found in config file"<<endl;
				return false;
			}
			
			if(resolverConfig["dT"]){
				dT = resolverConfig["dT"].as<float>();
			}else{
				cout<<"dT not found in config file"<<endl;
				return false;
			}
			
			if(resolverConfig["threadsPerBlock"]){
				threadsPerBlock = resolverConfig["threadsPerBlock"].as<int>();
			}else{
				cout<<"threadsPerBlock not found in config file"<<endl;
				return false;
			}

			if(resolverConfig["a"]){
				a = resolverConfig["a"].as<float>();
			}else{
				cout<<"a not found in config file"<<endl;
				return false;
			}

			if(resolverConfig["Ca"]){
				Ca = resolverConfig["Ca"].as<float>();
			}else{
				cout<<"Ca not found in config file"<<endl;
				return false;
			}

			if(resolverConfig["CloudChance"]){
				cloudChance = resolverConfig["CloudChance"].as<float>();
			}else{
				cout<<"CloudChance not found in config file"<<endl;
				return false;
			}

			if(resolverConfig["CloudMassCoef"]){
				cloudMassCoef = resolverConfig["CloudMassCoef"].as<float>();
			}else{
				cout<<"CloudMassCoef not found in config file"<<endl;
				return false;
			}

			Ms = Mtot * Msf;
			Mdm = Mtot - Ms;
		}else{
			cout<<"ResolverSettings not found in config file"<<endl;
			return false;
		}


	}catch(YAML::Exception &e){
		cout<<e.what()<<endl;
		return false;
	}

	//	Map the OpenGL VBO containing particle positions to a cuda pointer
	cudaGLRegisterBufferObject(posVboID);
	cudaGLMapBufferObject( (void **)&d_positions, posVboID);

	//	Allocate memory for old velocities vector
	cudaMalloc((void **)&d_oldPositions, 3 * NUM_PARTICLES * sizeof(float));

	//	Allocate memory for velocities vector
	cudaMalloc((void **)&d_velocities, 3 * NUM_PARTICLES * sizeof(float));
	cudaMalloc((void **)&d_oldVelocities, 3 * NUM_PARTICLES * sizeof(float));

	//	Allocate memory for accelerations vector
	cudaMalloc((void **)&d_accelerations, 3 * NUM_PARTICLES * sizeof(float));
	cudaMalloc((void **)&d_oldAccelerations, 3 * NUM_PARTICLES * sizeof(float));

	//	Allocate memory for jerks vector
	cudaMalloc((void **)&d_jerks, 3 * NUM_PARTICLES * sizeof(float));
	cudaMalloc((void **)&d_oldJerks, 3 * NUM_PARTICLES * sizeof(float));

	//	Allocate memory for masses
	cudaMalloc((void **)&d_masses, NUM_PARTICLES * sizeof(float));

	//	Allocate memory for scale radii
	cudaMalloc((void **)&d_scaleRadii, NUM_PARTICLES * sizeof(float));

	genBodies(d_positions, d_velocities, d_masses, d_scaleRadii, NUM_PARTICLES, Ms, Rs, Mdm, Rdm, cloudChance, cloudMassCoef, a, Ca, threadsPerBlock);

	cudaGLUnmapBufferObject(posVboID);


	return true;
}

void CudaAllPairsResolver::advanceTimeStep() {

	cudaGLMapBufferObject( (void **)&d_positions, posVboID);

	moveBodiesByDT_Euler(d_positions, d_velocities, d_accelerations, d_masses, d_scaleRadii, dT, NUM_PARTICLES, Ms, Rs, Mdm, Rdm, threadsPerBlock);

	cudaGLUnmapBufferObject(posVboID);
}


CudaAllPairsResolver::~CudaAllPairsResolver(void){
	cudaFree(d_oldPositions);
	cudaFree(d_velocities);
	cudaFree(d_oldVelocities);
	cudaFree(d_accelerations);
	cudaFree(d_oldAccelerations);
	cudaFree(d_jerks);
	cudaFree(d_oldJerks);
	cudaFree(d_masses);
	cudaFree(d_scaleRadii);
}
