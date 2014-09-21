#include <GL\glew.h>

void genBodies(float *d_pos, float *d_vel, float *d_mas, float *d_rad, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm, float cloudChance, float cloudMassCoef, float a, float Ca, int blockSize);

void moveBodiesByDT_Euler(float *d_pos, float *d_vel, float *d_acc, float *d_mas, float *d_rad, float dT, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm, int blockSize);