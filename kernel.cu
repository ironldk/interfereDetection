#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <set>
#include <random>
#include "CudaBVH.cuh"
#include "Point.cuh"
#include "Shader.h"
#include <cuda_gl_interop.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// settings
const unsigned int SCR_WIDTH = 2250;
const unsigned int SCR_HEIGHT = 1000;

// camera
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float yaw = -90.0f;	// yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
float pitch = 0.0f;
float lastX = 800.0f / 2.0;
float lastY = 600.0 / 2.0;
float fov = 45.0f;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = static_cast<float>(2.5 * deltaTime);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 direction;
    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(direction);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
}

void initVBOfloat3(GLuint& iVAO, GLuint& iVBO, int iNbVertex, cudaGraphicsResource_t& oCudaRsc) {
    glGenVertexArrays(1, &iVAO);
    glGenBuffers(1, &iVBO);

    glBindVertexArray(iVAO);

    glBindBuffer(GL_ARRAY_BUFFER, iVBO);
    glBufferData(GL_ARRAY_BUFFER, iNbVertex*3*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(
        &oCudaRsc, iVBO, cudaGraphicsRegisterFlagsWriteDiscard);
    assert(cudaStatus == cudaSuccess);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

void initTBO(GLuint& iTBO, GLuint& iTEX, int iNbTriangle, cudaGraphicsResource_t& oCudaRsc) {
    glGenBuffers(1, &iTBO);
    glBindBuffer(GL_TEXTURE_BUFFER, iTBO);
    glBufferData(GL_TEXTURE_BUFFER, iNbTriangle*sizeof(int), nullptr, GL_DYNAMIC_DRAW);

    glGenTextures(1, &iTEX);
    glBindTexture(GL_TEXTURE_BUFFER, iTEX);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, iTBO);

    cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(
        &oCudaRsc, iTBO, cudaGraphicsRegisterFlagsWriteDiscard);
    assert(cudaStatus == cudaSuccess);
}

// 清理VBO资源
void cleanupVBO(GLuint& iVBO, cudaGraphicsResource_t& iCUDAresource) {
    cudaGraphicsUnregisterResource(iCUDAresource);
    glDeleteBuffers(1, &iVBO);
}

int main(int argc, char* argv[]) {
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Interfere Detection", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("shader.vs", "shader.fs");

    // load and create a texture 
    // -------------------------
    unsigned int texture1, texture2;
    // texture 1
    // ---------
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    unsigned char* data = stbi_load("container.jpg", &width, &height, &nrChannels, 0);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    // texture 2
    // ---------
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    data = stbi_load("awesomeface.png", &width, &height, &nrChannels, 0);
    if (data) {
        // note that the awesomeface.png has transparency and thus an alpha channel, so make sure to tell OpenGL the data type is of GL_RGBA
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    // -------------------------------------------------------------------------------------------
    ourShader.use();
    ourShader.setInt("texture1", 0);
    ourShader.setInt("texture2", 1);

    // pass projection matrix to shader (as projection matrix rarely changes there's no need to do this per frame)
    // -----------------------------------------------------------------------------------------------------------
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    ourShader.setMat4("projection", projection);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    
    int device = 0;
    cudaDeviceProp prop;
    // 获取当前设备的 ID
    cudaGetDevice(&device);
    // 获取设备属性
    cudaGetDeviceProperties(&prop, device);
    // 获取最大线程块大小
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    ObjLoader obj1("object1_debug.stl"), obj2("object2_debug.stl");

    GLuint VBO1, VBO2, VBOBox1, VBOBox2, VAO1, VAO2, VAOBox1, VAOBox2, TBO1, TBO2, TEX1, TEX2;
    cudaGraphicsResource_t cudaRsc1, cudaRsc2, cudaRscBox1, cudaRscBox2, cudaRscHit1, cudaRscHit2;
    initVBOfloat3(VAO1, VBO1, obj1._tris.size()*3, cudaRsc1);
    initVBOfloat3(VAO2, VBO2, obj2._tris.size()*3, cudaRsc2);
    initVBOfloat3(VAOBox1, VBOBox1, obj1._tris.size()*48, cudaRscBox1);
    initVBOfloat3(VAOBox2, VBOBox2, obj2._tris.size()*48, cudaRscBox2);
    initTBO(TBO1, TEX1, obj1._tris.size(), cudaRscHit1);
    initTBO(TBO2, TEX2, obj2._tris.size(), cudaRscHit2);
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window)) {
        // Map buffer object for writing from CUDA
        cudaGraphicsMapResources(1, &cudaRsc1, 0);
        cudaGraphicsMapResources(1, &cudaRsc2, 0);
        cudaGraphicsMapResources(1, &cudaRscBox1, 0);
        cudaGraphicsMapResources(1, &cudaRscBox2, 0);
        cudaGraphicsMapResources(1, &cudaRscHit1, 0);
        cudaGraphicsMapResources(1, &cudaRscHit2, 0);
        float3 *pVertTri1 = nullptr, *pVertTri2 = nullptr, *pVertBox1 = nullptr, *pVertBox2 = nullptr;
		int *pHit1 = nullptr, *pHit2 = nullptr;
        size_t sizeTri1=0, sizeTri2=0, sizeBox1=0, sizeBox2=0, sizeHit1=0, sizeHit2=0;
        cudaGraphicsResourceGetMappedPointer((void**)&pVertTri1, &sizeTri1, cudaRsc1);
        cudaGraphicsResourceGetMappedPointer((void**)&pVertTri2, &sizeTri2, cudaRsc2);
        cudaGraphicsResourceGetMappedPointer((void**)&pVertBox1, &sizeBox1, cudaRscBox1);
        cudaGraphicsResourceGetMappedPointer((void**)&pVertBox2, &sizeBox2, cudaRscBox2);
        cudaGraphicsResourceGetMappedPointer((void**)&pHit1, &sizeHit1, cudaRscHit1);
        cudaGraphicsResourceGetMappedPointer((void**)&pHit2, &sizeHit2, cudaRscHit2);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        //ourShader.setMat4("model", model);
        CudaBVH* myBVH  = new CudaBVH(&obj1._tris, obj1._tris.size(), maxThreadsPerBlock, glm::value_ptr(model), pVertTri1, pVertBox1);
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        //ourShader.setMat4("model", model);
        CudaBVH* myBVH2 = new CudaBVH(&obj2._tris, obj2._tris.size(), maxThreadsPerBlock, glm::value_ptr(model), pVertTri2, pVertBox2);
        
        myBVH->boxIntersect(myBVH2->_SampleSize, myBVH2->d_BBoxLeaf, myBVH2->d_mesh, pHit2, pHit1);
        cudaDeviceSynchronize();







		//int* h_hit1 = new int[myBVH->_SampleSize];
		//int* h_hit2 = new int[myBVH2->_SampleSize];
		//cudaMemcpy(h_hit1, pHit1, myBVH->_SampleSize * sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_hit2, pHit2, myBVH2->_SampleSize * sizeof(int), cudaMemcpyDeviceToHost);
  //      cudaDeviceSynchronize();
  //      for(int i = 0; i < myBVH->_SampleSize; i++) {
  //          std::cout << "h_hit1[" << i << "]=" << h_hit1[i] << std::endl;
  //      }
  //      for (int i = 0; i < myBVH2->_SampleSize; i++) {
  //          std::cout << "h_hit2[" << i << "]=" << h_hit2[i] << std::endl;
  //      }
		//delete[] h_hit1;
		//delete[] h_hit2;
		//h_hit1 = nullptr;
		//h_hit2 = nullptr;






        cudaGraphicsUnmapResources(1, &cudaRsc1, 0);
        cudaGraphicsUnmapResources(1, &cudaRsc2, 0);
        cudaGraphicsUnmapResources(1, &cudaRscBox1, 0);
        cudaGraphicsUnmapResources(1, &cudaRscBox2, 0);
        cudaGraphicsUnmapResources(1, &cudaRscHit1, 0);
        cudaGraphicsUnmapResources(1, &cudaRscHit2, 0);






//// 在 boxIntersect 调用后添加调试
//int* debug_data1 = new int[myBVH->_SampleSize];
//int* debug_data2 = new int[myBVH2->_SampleSize];
//
//glBindBuffer(GL_TEXTURE_BUFFER, TBO1);
//glGetBufferSubData(GL_TEXTURE_BUFFER, 0, myBVH->_SampleSize * sizeof(int), debug_data1);
//printf("TBO1 data: %d %d\n", debug_data1[0], debug_data1[1]);
//
//glBindBuffer(GL_TEXTURE_BUFFER, TBO2);
//glGetBufferSubData(GL_TEXTURE_BUFFER, 0, myBVH2->_SampleSize * sizeof(int), debug_data2);
//printf("TBO2 data: %d %d\n", debug_data2[0], debug_data2[1]);
//
//delete[] debug_data1;
//delete[] debug_data2;
//debug_data1 = nullptr;
//debug_data2 = nullptr;







        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        // activate shader
        ourShader.use();

        // pass projection matrix to shader (note that in this case it could change every frame)
        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        ourShader.setMat4("projection", projection);

        // camera/view transformation
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        ourShader.setMat4("view", view);

        // 渲染第一个物体
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER, TEX1);
        ourShader.setInt("triHitBuffer", 2);

        glBindVertexArray(VAO1);
        // 计算三角形数量并绘制
        int triangleCount1 = myBVH->_SampleSize;
		ourShader.setFloat4("color", 1.0f, 0.0f, 0.0f, 1.0f);
		ourShader.setInt("nbVertexPerElement", 3);
        glDrawArrays(GL_TRIANGLES, 0, triangleCount1 * 3);
        
        glBindVertexArray(VAOBox1);
		ourShader.setFloat4("color", 0.0f, 0.0f, 1.0f, 0.2f);
        ourShader.setInt("nbVertexPerElement", 24);
        //glDrawArrays(GL_LINES, 0, triangleCount1 * 48);
        glDrawArrays(GL_LINES, 0, triangleCount1 * 24);

        // 渲染第二个物体（稍微偏移以避免重叠）
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER, TEX2);
        ourShader.setInt("triHitBuffer", 3);

        glBindVertexArray(VAO2);
        
        int triangleCount2 = myBVH2->_SampleSize;
        ourShader.setFloat4("color", 1.0f, 0.0f, 0.0f, 1.0f);
        ourShader.setInt("nbVertexPerElement", 3);
        glDrawArrays(GL_TRIANGLES, 0, triangleCount2 * 3);

        glBindVertexArray(VAOBox2);
        ourShader.setFloat4("color", 0.0f, 0.0f, 1.0f, 0.2f);
        ourShader.setInt("nbVertexPerElement", 24);
        //glDrawArrays(GL_LINES, 0, triangleCount2 * 48);
        glDrawArrays(GL_LINES, 0, triangleCount2 * 24);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        delete (myBVH); delete (myBVH2);
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO1);
    glDeleteVertexArrays(1, &VAO2);
    glDeleteVertexArrays(1, &VAOBox1);
    glDeleteVertexArrays(1, &VAOBox2);
    glDeleteTextures(1, &TEX1);
    glDeleteTextures(1, &TEX2);
    glDeleteTextures(1, &texture1);
    glDeleteTextures(1, &texture2);
    
    // 清理VBO资源
    cleanupVBO(VBO1, cudaRsc1);
    cleanupVBO(VBO2, cudaRsc2);
    cleanupVBO(VBOBox1, cudaRscBox1);
    cleanupVBO(VBOBox2, cudaRscBox2);
	glDeleteBuffers(1, &TBO1);
	glDeleteBuffers(1, &TBO2);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();

	checkCudaErrors(cudaDeviceReset());
	return 0;
}