#version 330 core
out vec4 FragColor;

flat in int triID;

// texture samplers
uniform sampler2D texture1;
uniform sampler2D texture2;

uniform vec4 color;
uniform isamplerBuffer triHitBuffer;

void main() {
	// linearly interpolate between both textures (80% container, 20% awesomeface)
	//FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);

	int hit = texelFetch(triHitBuffer, triID).r;
	if (hit == 1){
		FragColor = vec4(1.0, 1.0, 1.0, color.a); // White for hit
		//FragColor = color;
	} else {
		FragColor = color;
	}
}