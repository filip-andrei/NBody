#include <GL\glew.h>

void genBodies(float *d_pos, float *d_vel, float *d_mas, float *d_rad, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm, float cloudChance, float cloudMassCoef, int blockSize);
void moveBodiesByDT(float *d_pos, float *d_vel, float *d_mas, float dT, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm, int blockSize);