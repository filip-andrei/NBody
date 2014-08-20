#pragma once

#include <GL\glew.h>

class INbodyResolver
{
public:

	virtual void advanceTimeStep()=0;

	virtual void loadSimConfig()=0;
	virtual void setPosBufferID(GLuint vboID)=0;

	virtual void initialize()=0;

	virtual ~INbodyResolver(void){ };
};

