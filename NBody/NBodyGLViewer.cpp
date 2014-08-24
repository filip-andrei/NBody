#define NOMINMAX

#include "NBodyGLViewer.h"

#include "lodepng.h"

#include <glm\gtx\transform.hpp>

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <time.h>


void NBodyGLViewer::initGL(int *argc, char **argv){
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(MAX_WIDTH, MAX_HEIGHT);
	glutCreateWindow(WINDOW_TITLE);

	glewInit();
}


GLuint NBodyGLViewer::loadShaders(const char *vertex_file_path, const char *fragment_file_path){
	// Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
 
    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);

    if(VertexShaderStream.is_open())
    {
        std::string Line = "";
        while(getline(VertexShaderStream, Line)){
            VertexShaderCode += "\n" + Line;
		}
        VertexShaderStream.close();
    }
 
    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if(FragmentShaderStream.is_open()){
        std::string Line = "";
        while(getline(FragmentShaderStream, Line))
            FragmentShaderCode += "\n" + Line;
        FragmentShaderStream.close();
    }
 
    GLint Result = GL_FALSE;
    int InfoLogLength;
 
    // Compile Vertex Shader
    printf("Compiling shader : %s\n", vertex_file_path);
    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);
 
    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> VertexShaderErrorMessage(InfoLogLength);
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
 
    // Compile Fragment Shader
    printf("Compiling shader : %s\n", fragment_file_path);
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);
 
    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> FragmentShaderErrorMessage(InfoLogLength);
    glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);
 
    // Link the program
    fprintf(stdout, "Linking program\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);
 
    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> ProgramErrorMessage( std::max(InfoLogLength, 1) );
    glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
    fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
 
    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);
 
    return ProgramID;
}

void windowResize( GLint newWidth, GLint newHeight ) {
	NBodyGLViewer &viewer = NBodyGLViewer::getInstance();

	glViewport( 0, 0, newWidth, newHeight );

	viewer.MAX_WIDTH = newWidth;
	viewer.MAX_HEIGHT = newHeight;
}

void render(){
	NBodyGLViewer &viewer = NBodyGLViewer::getInstance();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPointSize(viewer.POINT_SIZE);

	glm::mat4 model = glm::scale(glm::vec3(viewer.viewData.zoom));
	model *= glm::rotate(model, viewer.viewData.thetaY, glm::vec3(1,0,0));
	model *= glm::rotate(model, viewer.viewData.thetaX, glm::vec3(0,0,1));

	glm::mat4 view = glm::lookAt(glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,0.0,0.0), glm::vec3(0.0,1.0,0.0));

	glm::mat4 projection = glm::perspective(45.0f, (float)viewer.MAX_WIDTH/viewer.MAX_HEIGHT, 0.1f, 100.0f);

	viewer.MVP = projection * view * model;

	glUseProgram(viewer.shaderProgramID);
	glUniformMatrix4fv(viewer.matrixID, 1, GL_FALSE, &viewer.MVP[0][0]);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, viewer.posArray);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		0
	);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_POINTS, 0, viewer.NUM_PARTICLES);

	glDisableVertexAttribArray(0);

	glutSwapBuffers();
}

void mouseBtnDown(int button, int state, int x, int y){
	NBodyGLViewer &viewer = NBodyGLViewer::getInstance();

	if(state == GLUT_DOWN){
		viewer.mouseLoc.x = x;
		viewer.mouseLoc.y = y;
	}

	if(button == 3){
		viewer.viewData.zoom += 0.05f * viewer.viewData.zoom;
	}
	else if(button == 4){
		viewer.viewData.zoom -= 0.05f * viewer.viewData.zoom;
	}
}

void mouseMove(int x, int y){
	NBodyGLViewer &viewer = NBodyGLViewer::getInstance();

	viewer.viewData.thetaX += (x - viewer.mouseLoc.x) * 0.5f;
	viewer.viewData.thetaY += (y - viewer.mouseLoc.y) * 0.5f;


	viewer.mouseLoc.x = x;
	viewer.mouseLoc.y = y;
}

void timer(int flag) {

	NBodyGLViewer &viewer = NBodyGLViewer::getInstance();
    
	glutPostRedisplay();

	viewer.resolver->advanceTimeStep();

	if(viewer.exportFrames){
		viewer.exportFrame();
	}

	int newElapsedTime = glutGet(GLUT_ELAPSED_TIME);

	char title[512];
	sprintf_s(title, 512 * sizeof(char), "%s - %d ms per frame", viewer.WINDOW_TITLE, (int)(newElapsedTime - viewer.elapsedTime));
	viewer.elapsedTime = glutGet(GLUT_ELAPSED_TIME);
	glutSetWindowTitle(title);

	float delayToNextFrame =  (CLOCKS_PER_SEC/viewer.MAX_FPS) - (newElapsedTime-viewer.elapsedTime);
	delayToNextFrame = floor(delayToNextFrame+0.5);
	delayToNextFrame < 0 ? delayToNextFrame = 0 : NULL;

	glutTimerFunc(delayToNextFrame, timer, 0);
}

void NBodyGLViewer::registerCallbacks(){

	glutDisplayFunc(render);
	glutReshapeFunc(windowResize);
	glutMouseFunc(mouseBtnDown);
	glutMotionFunc(mouseMove);
	glutTimerFunc(30, timer, 0);
}

void NBodyGLViewer::exportFrame(){
	char filename[1024];
	sprintf(filename, "D:\\Temp\\Frames\\110k-Msf0.14-SPa0.6-%05d.png", frameCounter++);
	
	glReadPixels(0, 0, MAX_WIDTH, MAX_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);
	
	printf("Saving frame to %s\n", filename);
	unsigned int error = lodepng_encode24_file(filename, frameBuffer, MAX_WIDTH, MAX_HEIGHT);

	if(error){
		printf("error occurred on frame export %u: %s\n", error, lodepng_error_text(error));
	}
}





void NBodyGLViewer::init(int *argc, char **argv, AbstractResolver *resolver){
	this->resolver = resolver;

	//	Window Config
	WINDOW_TITLE = "N-Body Window";
	MAX_WIDTH = 1280;
	MAX_HEIGHT = 720;
	MAX_FPS = 60;
	POINT_SIZE = 1.0f;
	ZOOM = 15.0f;
	exportFrames = false;

	//	Sim Config
	NUM_PARTICLES = 256 * 430;


	initGL(argc, argv);

	if(exportFrames){
		frameBuffer = (unsigned char *)malloc(3 * sizeof(unsigned char) * MAX_WIDTH * MAX_HEIGHT);
	}

	viewData.thetaX = 0.0f;
	viewData.thetaY = 0.0f;
	viewData.zoom = 1.0f / ZOOM;
	mouseLoc.x = 0;
	mouseLoc.y = 0;


	shaderProgramID = loadShaders("vertexShader.glsl", "fragmentShader.glsl");

	glGenBuffers(1, &posArray);
	glBindBuffer(GL_ARRAY_BUFFER, posArray);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * NUM_PARTICLES, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	
	glm::mat4 model = glm::scale(glm::vec3(1.0));
	glm::mat4 view = glm::lookAt(glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,0.0,0.0), glm::vec3(0.0,1.0,0.0));
	glm::mat4 projection = glm::perspective(45.0f, (float)MAX_WIDTH/MAX_HEIGHT, 0.1f, 100.0f);

	MVP = projection * view * model;

	matrixID = glGetUniformLocation(shaderProgramID, "MVP");


	resolver->loadSimConfig();
	resolver->setPosBufferID(posArray);
	resolver->initialize();

	registerCallbacks();

	elapsedTime = glutGet(GLUT_ELAPSED_TIME);
}

void NBodyGLViewer::start(){
	glutMainLoop();
}

NBodyGLViewer::~NBodyGLViewer(void)
{
}
