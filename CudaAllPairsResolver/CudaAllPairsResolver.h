#pragma once
#include "..\AbstractResolver\AbstractResolver.h"


class CudaAllPairsResolver :
	public AbstractResolver
{
private:


	//	Total number of bodies in simulation
	int NUM_PARTICLES;

	//	Total Mass of the galaxy, disk and dark halo (SM)
	float Mtot;

	//	Fraction of the total mass belonging to the galactic disk (regular matter)
	float Msf;

	//	Scale radius for stellar density distribution (Pcs)
	float Rs;

	//	Scale radius for dark matter density distribution (Pcs)
	float Rdm;
	
	//	Time increment in each simulation step (Myr)
	float dT;

	//	Total mass of dark matter in galaxy (SM)
	float Mdm;

	//	Total stellar mass in galaxy
	float Ms;

	//	Scale radius for particle gravitational damping at close distances (Pcs)
	float a;

	//	Same as above only for molecular clouds (Pcs)
	float Ca;

	//	Ratio of molecular clouds to regular stars
	float cloudChance;

	//	Times the mass of a molecular cloud is greater than a regular star
	float cloudMassCoef;

	//	Id of the GL Vertex Buffer containing positional data
	GLuint posVboID;

	//	Cuda pointer to positional vector data (PCs)
	float *d_positions;

	//	Cuda pointer to velocity vector data (Km/s)
	float *d_velocities;

	//	Cuda pointer to mass data (SM)
	float *d_masses;

	//	Cuda pointer to gravitational damping scale radii (Pcs)
	float *d_scaleRadii;

	//	Number of threads in a thread block
	int threadsPerBlock;

public:
	CudaAllPairsResolver(void);

	void advanceTimeStep() override;

	void setPosBufferID(GLuint vboID) override;

	bool initialize(YAML::Node &config) override;

	~CudaAllPairsResolver(void);
};

