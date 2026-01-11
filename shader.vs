#version 330 core
layout (location = 0) in vec3 aPos;

flat out int triID;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int nbVertexPerElement;

void main() {
	//gl_Position = projection * view * model * vec4(aPos, 1.0f);
	gl_Position = projection * view * vec4(aPos, 1.0f);
	triID = gl_VertexID / nbVertexPerElement;
}