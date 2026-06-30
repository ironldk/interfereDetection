#version 330 core
out vec4 FragColor;

flat in int triID;

// texture samplers
uniform sampler2D texture1;
uniform sampler2D texture2;

uniform vec4 color;
uniform vec4 colorHit;
uniform isamplerBuffer triHitBuffer;

void main() {
	// linearly interpolate between both textures (80% container, 20% awesomeface)
	//FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);

	int hit = texelFetch(triHitBuffer, triID).r;
	vec4 arrColor[2];
	arrColor[0]= color;
	arrColor[1]= colorHit;
	FragColor = arrColor[hit];
}