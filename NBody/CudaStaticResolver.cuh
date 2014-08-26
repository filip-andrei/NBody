#include <GL\glew.h>

void genBodies(float *d_pos, float *d_vel, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm, int blockSize);
void moveBodiesByDT_staticPotential(float *d_pos, float *d_vel, float dT, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm, int blockSize);