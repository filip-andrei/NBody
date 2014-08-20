//	OpenGL lib

#include <GL\glew.h>
#include <GL\wglew.h>
#include <GL\freeglut.h>

// GLM
#include <glm\glm.hpp>
#include <glm\gtx\transform.hpp>

//	LodePNG for exporting frames
#include "lodepng.h"

//	Various
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "shader.h"
//#include "cuda_kernel.cuh"

#include "CudaAllPairsStaticResolver.cuh"

//	Perspective stuff
struct ViewData{
	float zoom;
	float thetaX;
	float thetaY;
} viewData;


//	Window Config
const char *WINDOW_TITLE = "N-Body Window";
const int MAX_WIDTH = 1280;
const int MAX_HEIGHT = 720;
const int MAX_FPS = 60;
const float POINT_SIZE = 1.0f;
const float ZOOM = 15.0f;
const bool exportFrames = false;

//	Sim Config
const int NUM_PARTICLES = 256 * 4300;


//	Various
GLuint posArray;									//	Array in gpu memory containing (x,y,z) positional information	(Pcs)
GLuint velArray;									//	Array in gpu memory containing (x,y,z) velocity information		(Km/s)
GLuint shaderProgramID;
GLuint matrixID;
glm::mat4 MVP;
POINT mouseLoc;

unsigned int elapsedTime;

//	Buffer for saving frame pixels
unsigned char *frameBuffer;
int frameCounter = 0;










INbodyResolver *resolver;






void initOpenGL(int *argc, char **argv){
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(MAX_WIDTH, MAX_HEIGHT);
	glutCreateWindow(WINDOW_TITLE);

	glewInit();
}

void init(){
	if(exportFrames){
		frameBuffer = (unsigned char *)malloc(3 * sizeof(unsigned char) * MAX_WIDTH * MAX_HEIGHT);
	}

	viewData.thetaX = 0.0f;
	viewData.thetaY = 0.0f;
	viewData.zoom = 1.0f / ZOOM;
	mouseLoc.x = 0;
	mouseLoc.y = 0;


	shaderProgramID = LoadShaders("vertexShader.glsl", "fragmentShader.glsl");
	

	glGenBuffers(1, &posArray);
	glBindBuffer(GL_ARRAY_BUFFER, posArray);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * NUM_PARTICLES, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &velArray);
	glBindBuffer(GL_ARRAY_BUFFER, velArray);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * NUM_PARTICLES, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	
	//genBodies(posArray, velArray, NUM_PARTICLES, Mtot * Msf, Rs, Mtot * (1.0f - Msf), Rdm);
	
	
	glm::mat4 model = glm::scale(glm::vec3(1.0));
	glm::mat4 view = glm::lookAt(glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,0.0,0.0), glm::vec3(0.0,1.0,0.0));
	glm::mat4 projection = glm::perspective(45.0f, (float)MAX_WIDTH/MAX_HEIGHT, 0.1f, 100.0f);

	MVP = projection * view * model;

	matrixID = glGetUniformLocation(shaderProgramID, "MVP");


	elapsedTime = glutGet(GLUT_ELAPSED_TIME);
}



void renderScene(){

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPointSize(POINT_SIZE);

	glm::mat4 model = glm::scale(glm::vec3(viewData.zoom));
	model *= glm::rotate(model, viewData.thetaY, glm::vec3(1,0,0));
	model *= glm::rotate(model, viewData.thetaX, glm::vec3(0,0,1));

	glm::mat4 view = glm::lookAt(glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,0.0,0.0), glm::vec3(0.0,1.0,0.0));

	glm::mat4 projection = glm::perspective(45.0f, (float)MAX_WIDTH/MAX_HEIGHT, 0.1f, 100.0f);

	MVP = projection * view * model;

	glUseProgram(shaderProgramID);
	glUniformMatrix4fv(matrixID, 1, GL_FALSE, &MVP[0][0]);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, posArray);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		0
	);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

	glDisableVertexAttribArray(0);

	glutSwapBuffers();
}

void exportFrame(){
	char filename[1024];
	sprintf(filename, "D:\\Temp\\Frames\\110k-Msf0.14-SPa0.6-%05d.png", frameCounter++);
	
	glReadPixels(0, 0, MAX_WIDTH, MAX_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);
	
	printf("Saving frame to %s\n", filename);
	unsigned int error = lodepng_encode24_file(filename, frameBuffer, MAX_WIDTH, MAX_HEIGHT);

	if(error){
		printf("error occurred on frame export %u: %s\n", error, lodepng_error_text(error));
	}
}

void timer(int flag) {
    
	glutPostRedisplay();
	//moveBodiesByDT_NBody(posArray, velArray, dT, MPart, Mtot * (1.0f - Msf), Rdm, NUM_PARTICLES);
	//moveBodiesByDT_staticPotential(posArray, velArray, dT, MPart, NUM_PARTICLES, Mtot*Msf, Rs, Mtot*(1.0f-Msf), Rdm);

	resolver->advanceTimeStep();

	if(exportFrames){
		exportFrame();
	}

	int newElapsedTime = glutGet(GLUT_ELAPSED_TIME);

	char title[512];
	sprintf_s(title, 512 * sizeof(char), "%s - %d ms per frame", WINDOW_TITLE, (int)(newElapsedTime - elapsedTime));
	elapsedTime = glutGet(GLUT_ELAPSED_TIME);
	glutSetWindowTitle(title);

	float delayToNextFrame =  (CLOCKS_PER_SEC/MAX_FPS) - (newElapsedTime-elapsedTime);
	delayToNextFrame = floor(delayToNextFrame+0.5);
	delayToNextFrame < 0 ? delayToNextFrame = 0 : NULL;

	glutTimerFunc(delayToNextFrame, timer, 0);
}

void mouseBtnDown(int button, int state, int x, int y){
	if(state == GLUT_DOWN){
		mouseLoc.x = x;
		mouseLoc.y = y;
	}

	if(button == 3){
		viewData.zoom += 0.05f * viewData.zoom;
	}
	else if(button == 4){
		viewData.zoom -= 0.05f * viewData.zoom;
	}
}

void mouseMove(int x, int y){
	viewData.thetaX += (x - mouseLoc.x) * 0.5f;
	viewData.thetaY += (y - mouseLoc.y) * 0.5f;


	mouseLoc.x = x;
	mouseLoc.y = y;
}

//void cleanup(){
//	printf("Doing cleanup\n");
//
//	
//	glDeleteBuffers(1, &posArray);
//	glDeleteBuffers(1, &velArray);
//
//
//	glDeleteProgram(programID);
//
//	if(exportFrames){
//		free(frameBuffer);
//	}
//}

int main(int argc, char **argv){
	
	//	Initialize GL subsystem
	initOpenGL(&argc, argv);
	init();


	resolver = new CudaAllPairsStaticResolver();
	resolver->loadSimConfig();
	resolver->setPosBufferID(posArray);
	resolver->initialize();


	//	Register callback functions
	glutDisplayFunc(renderScene);
	glutMouseFunc(mouseBtnDown);
	glutMotionFunc(mouseMove);
	glutTimerFunc(30, timer, 0);
	glutMainLoop();

	return 1;
}
