#include "CPUAllPairsResolver.h"
#include "constants.h"

#include <iostream>
#include <stdlib.h>
#include <random>


#ifdef _DEBUG
#pragma comment(lib, "libyaml-cppmdd.lib")
#else
#pragma comment(lib, "libyaml-cppmd.lib")
#endif




using namespace std;


struct Vec3f{
	float x;
	float y;
	float z;
};


float CPUAllPairsResolver::mbessi0(float x) {
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

float CPUAllPairsResolver::mbessi1(float x) {
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

float CPUAllPairsResolver::mbessk0(float x) {
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

float CPUAllPairsResolver::mbessk1(float x) {
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



float CPUAllPairsResolver::dmMassAtRadius(float r){
	return (Mdm * r * r) / powf(r + Rdm, 2);
}



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


		//	Set positions
		float rx = -Rs * log(1.0f - x);

		float Sz = -(1.0f/2.0f) * (0.1f * Rs) * log(-((z-1)/z));		
		float Sx = sqrt(rx*rx) * cos(2.0f * 3.1416f * y);
		float Sy = sqrt(rx*rx) * sin(2.0f * 3.1416f * y);


		positions[i * 3] = Sx;
		positions[i * 3 + 1] = Sy;
		positions[i * 3 + 2] = Sz;
		//	---

		//	Set velocities

		float realRad = sqrt(Sx * Sx + Sy * Sy + Sz * Sz);

		float t = realRad / (2.0f * Rs);
		float absVel = sqrt( (G * dmMassAtRadius(realRad)) / realRad + ((2.0f * G * Ms) / Rs) * t * t * (mbessi0(t)*mbessk0(t) - mbessi1(t)*mbessk1(t)) );
		
		Vec3f velUnitVector;
		velUnitVector.x = - Sy / sqrt(Sx*Sx+Sy*Sy);
		velUnitVector.y = Sx / sqrt(Sx*Sx+Sy*Sy);
		velUnitVector.z = 0;

		velocities[i * 3] = velUnitVector.x * absVel;
		velocities[i * 3 + 1] = velUnitVector.y * absVel;
		velocities[i * 3 + 2] = velUnitVector.z * absVel;
	}

	glBindBuffer(GL_ARRAY_BUFFER, posVboID);
	glBufferData(GL_ARRAY_BUFFER, 3 * NUM_PARTICLES * sizeof(float), positions, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}


void CPUAllPairsResolver::advanceTimeStep(){

	for (int i = 0; i < NUM_PARTICLES; i++){

		Vec3f currentVel;
		currentVel.x = velocities[i * 3];
		currentVel.y = velocities[i * 3 + 1];
		currentVel.z = velocities[i * 3 + 2];

		Vec3f currentPos;
		currentPos.x = positions[i * 3] + (currentVel.x * velConvFactor * dT);
		currentPos.y = positions[i * 3 + 1] + (currentVel.y * velConvFactor * dT);
		currentPos.z = positions[i * 3 + 2] + (currentVel.z * velConvFactor * dT);

		Vec3f totalAcc;
		totalAcc.x = 0.0f;
		totalAcc.y = 0.0f;
		totalAcc.z = 0.0f;

		//	Add acceleration from stellar sources
		for(int j = 0; j < NUM_PARTICLES; j++){

			if(i == j)continue;

			Vec3f destParticlePos;
			destParticlePos.x = positions[j * 3];
			destParticlePos.y = positions[j * 3 + 1];
			destParticlePos.z = positions[j * 3 + 2];

			float destParticleMass = Ms / NUM_PARTICLES;

			Vec3f rVec;
			rVec.x = currentPos.x - destParticlePos.x;
			rVec.y = currentPos.y - destParticlePos.y;
			rVec.z = currentPos.z - destParticlePos.z;

			float r = sqrtf(rVec.x * rVec.x + rVec.y * rVec.y + rVec.z * rVec.z);
			
			Vec3f rUnit;
			rUnit.x = rVec.x / r;
			rUnit.y = rVec.y / r;
			rUnit.z = rVec.z / r;

			float acc = -((G * destParticleMass * r) / ( sqrtf(powf(r*r + a * a, 3.0f)) )) * (1 / kmPerPc);

			totalAcc.x += acc * rUnit.x;
			totalAcc.y += acc * rUnit.y;
			totalAcc.z += acc * rUnit.z;
		}
		//	---

		//	Add acceleration from dark matter

		Vec3f rVec = currentPos;
		float r = sqrtf(rVec.x * rVec.x + rVec.y * rVec.y + rVec.z * rVec.z);
		Vec3f rUnit;
		rUnit.x = rVec.x / r;
		rUnit.y = rVec.y / r;
		rUnit.z = rVec.z / r;

		float relevantDmMass = dmMassAtRadius(r);

		float acc = -((G * relevantDmMass * r) / ( sqrtf(powf(r*r + a * a, 3.0f)) )) * (1 / kmPerPc);

		totalAcc.x += acc * rUnit.x;
		totalAcc.y += acc * rUnit.y;
		totalAcc.z += acc * rUnit.z;

		//	---

		currentVel.x += totalAcc.x * (dT * secPerMYr);
		currentVel.y += totalAcc.y * (dT * secPerMYr);
		currentVel.z += totalAcc.z * (dT * secPerMYr);

		positions[i * 3] = currentPos.x;
		positions[i * 3 + 1] = currentPos.y;
		positions[i * 3 + 2] = currentPos.z;

		velocities[i * 3] = currentVel.x;
		velocities[i * 3 + 1] = currentVel.y;
		velocities[i * 3 + 2] = currentVel.z;
	}

	//	Update vertex buffer
	glBindBuffer(GL_ARRAY_BUFFER, posVboID);
	glBufferData(GL_ARRAY_BUFFER, 3 * NUM_PARTICLES * sizeof(float), positions, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

CPUAllPairsResolver::~CPUAllPairsResolver(void)
{
	delete[] positions;
	delete[] velocities;
}
