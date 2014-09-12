#include "CPUAllPairsResolver.h"

#include <iostream>
#include <stdlib.h>
#include <random>


#ifdef _DEBUG
#pragma comment(lib, "libyaml-cppmdd.lib")
#else
#pragma comment(lib, "libyaml-cppmd.lib")
#endif




using namespace std;


CPUAllPairsResolver::CPUAllPairsResolver(void)
{

}

void CPUAllPairsResolver::setPosBufferID(GLuint vboID){
	this->posVboID = vboID;
}

bool CPUAllPairsResolver::initialize(YAML::Node &config){
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


	positions = new float[NUM_PARTICLES * 3];
	velocities = new float[NUM_PARTICLES * 3];
	masses = new float[NUM_PARTICLES];
	scaleRadii = new float[NUM_PARTICLES];
	


	genBodies();

	

	return true;
}


void CPUAllPairsResolver::genBodies(){

	const float HI = 0.0f;
	const float LO = 1.0f;

	std::default_random_engine gen;
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	
	for(int i = 0; i < NUM_PARTICLES; i++){
		
		float x = dist(gen);
		float y = dist(gen);
		float z = dist(gen);
		float w = dist(gen);

		if(x == 0.0f || x == 1.0f || y == 0.0f || y == 1.0f){
			i--;
			continue;
		}

		float rx = -Rs * log(1.0f - x);

		float Sz = -(1.0f/2.0f) * (0.1f * Rs) * log(-((z-1)/z));		
		float Sx = sqrt(rx*rx) * cos(2.0f * 3.1416f * y);
		float Sy = sqrt(rx*rx) * sin(2.0f * 3.1416f * y);


		positions[i * 3] = Sx;
		positions[i * 3 + 1] = Sy;
		positions[i * 3 + 2] = Sz;


	}

	glBindBuffer(GL_ARRAY_BUFFER, posVboID);
	glBufferData(GL_ARRAY_BUFFER, 3 * NUM_PARTICLES * sizeof(float), positions, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}


void CPUAllPairsResolver::advanceTimeStep(){

}

CPUAllPairsResolver::~CPUAllPairsResolver(void)
{
	delete[] positions;
	delete[] velocities;
	delete[] masses;
	delete[] scaleRadii;
}
