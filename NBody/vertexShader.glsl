#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
uniform mat4 MVP;

out float opacity;

float densityAtRadius(float r){
	return (13282.15f / (2 * 3.14159f * 3160.0f * 3160.0f)) * exp(-r/3160.0f);
}

void main(){
	vec4 v = vec4(vertexPosition_modelspace, 1.0f);

	gl_Position = MVP * v;

	

	//opacity = 1.0f - (densityAtRadius(length(vec2(v))) / densityAtRadius(0.0f));
	opacity = 1.0f;
}