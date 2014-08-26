#include "CudaStaticResolver.h"
#include "CudaStaticResolver.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>


using namespace std;


CudaStaticResolver::CudaStaticResolver(void)
{
	NUM_PARTICLES = -1;

	Mtot = -1;
	Msf = -1;

	Ms = -1;
	Mdm = -1;

	Rs = -1;
	Rdm = -1;

	dT = -1;

	threadsPerBlock = -1;
}


void CudaStaticResolver::setPosBufferID(GLuint vboID){
	posVboID = vboID;
}

bool CudaStaticResolver::initialize(YAML::Node &config){
	
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

	//	Allocate memory for velocities vector
	cudaMalloc((void **)&d_velocities, 3 * NUM_PARTICLES * sizeof(float));

	genBodies(d_positions, d_velocities, NUM_PARTICLES, Ms, Rs, Mdm, Rdm, threadsPerBlock);

	cudaGLUnmapBufferObject(posVboID);


	return true;
}

void CudaStaticResolver::advanceTimeStep() {

	cudaGLMapBufferObject( (void **)&d_positions, posVboID);

	moveBodiesByDT_staticPotential(d_positions, d_velocities, dT, NUM_PARTICLES, Ms, Rs, Mdm, Rdm, threadsPerBlock);

	cudaGLUnmapBufferObject(posVboID);
}


CudaStaticResolver::~CudaStaticResolver(void){
	cudaFree(d_positions);
	cudaFree(d_velocities);
}
