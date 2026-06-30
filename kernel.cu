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

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

class Application {
private:
    // settings
    unsigned int _scrWidth = 2250;
    unsigned int _scrHeight = 1000;

    // camera
    glm::vec3 _cameraPos = glm::vec3(4.618802f, 4.0f, 24.0f);
    glm::vec3 _cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 _cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    bool _firstMouse = true;
    float _yaw = -90.0f;	// yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
    float _pitch = 0.0f;
    float _lastX = 800.0f / 2.0;
    float _lastY = 600.0 / 2.0;
    float _fov = 45.0f;

    // timing
    float _deltaTime = 0.0f;	// time between current frame and last frame
    float _lastFrame = 0.0f;

    // Model matrix parameters for object 2 (wrench) - adjustable via ImGui
    struct ModelParams {
        float translateX = 4.618802f;
        float translateY = 4.0f;
        float translateZ = 0.0f;
        float rotateAngle = -1.84f;
        float rotateX = 0.0f;
        float rotateY = 0.0f;
        float rotateZ = 1.0f;
        float scale = 1.75f;
    } _modelParams;

    // Resources for rendering
    GLFWwindow* _window = nullptr;
    Shader* _ourShader = nullptr;
    unsigned int _texture[2];
    float _colorObj[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    float _colorBox[4] = { 0.0f, 0.0f, 1.0f, 0.2f };
    float _colorHit[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

    ObjLoader* _obj[2] = { nullptr, nullptr };
    int _triNum[2];
    GLuint _VBO[2], _VBOBox[2], _VAO[2], _VAOBox[2], _TBO[2], _TEX[2];
    cudaGraphicsResource_t _cudaRsc[2], _cudaRscBox[2], _cudaRscHit[2];
    int _maxThreadsPerBlock = 0;

    bool _mouseCaptured = true;

    // Static callback wrappers that get Application instance from user pointer
    static void framebuffer_size_callback_wrapper(GLFWwindow* window, int width, int height) {
        Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
        app->framebufferSizeCallback(window, width, height);
    }

    static void mouse_callback_wrapper(GLFWwindow* window, double xpos, double ypos) {
        Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
        app->mouseCallback(window, xpos, ypos);
    }

    static void scroll_callback_wrapper(GLFWwindow* window, double xoffset, double yoffset) {
        Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
        app->scrollCallback(window, xoffset, yoffset);
    }

    static void window_refresh_callback_wrapper(GLFWwindow* window) {
        Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
        app->windowRefreshCallback(window);
    }

    // Callback implementations
    void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        _scrWidth = width;
        _scrHeight = height;
        glViewport(0, 0, width, height);
    }

    void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
        // Only process mouse movement when captured (not interacting with UI)
        if (!_mouseCaptured) {
            return;
        }
        
        if (_firstMouse) {
            _lastX = xpos;
            _lastY = ypos;
            _firstMouse = false;
        }

        float xoffset = xpos - _lastX;
        float yoffset = _lastY - ypos;
        _lastX = xpos;
        _lastY = ypos;

        float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        _yaw += xoffset;
        _pitch += yoffset;

        if (_pitch > 89.0f)
            _pitch = 89.0f;
        if (_pitch < -89.0f)
            _pitch = -89.0f;

        glm::vec3 direction;
        direction.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
        direction.y = sin(glm::radians(_pitch));
        direction.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));
        _cameraFront = glm::normalize(direction);
    }

    void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        _fov -= (float)yoffset;
        if (_fov < 1.0f)
            _fov = 1.0f;
        if (_fov > 45.0f)
            _fov = 45.0f;
    }

    void windowRefreshCallback(GLFWwindow* window) {
        render();
        glfwSwapBuffers(window);
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

    // process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
    // ---------------------------------------------------------------------------------------------------------
    void processInput(GLFWwindow* window) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Toggle mouse capture with F1 key
        static bool f1Pressed = false;
        if (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS) {
            if (!f1Pressed) {
                _mouseCaptured = !_mouseCaptured;
                glfwSetInputMode(window, GLFW_CURSOR, _mouseCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
                f1Pressed = true;
            }
        } else {
            f1Pressed = false;
        }

        // Only process camera movement when mouse is captured (not interacting with UI)
        if (_mouseCaptured) {
            float cameraSpeed = static_cast<float>(2.5 * _deltaTime);
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                _cameraPos += cameraSpeed * _cameraFront;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                _cameraPos -= cameraSpeed * _cameraFront;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                _cameraPos -= glm::normalize(glm::cross(_cameraFront, _cameraUp)) * cameraSpeed;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                _cameraPos += glm::normalize(glm::cross(_cameraFront, _cameraUp)) * cameraSpeed;
        }
    }

public:
    Application() = default;
    ~Application() {
        // Cleanup resources
        if (_window) {
            // optional: de-allocate all resources once they've outlived their purpose:
            // ------------------------------------------------------------------------
            for (int i = 0; i < 2; ++i) {
                if (_VAO[i]) {
                    glDeleteVertexArrays(1, &_VAO[i]);
                }
                if (_VAOBox[i]) {
                    glDeleteVertexArrays(1, &_VAOBox[i]);
                }
                if (_TEX[i]) {
                    glDeleteTextures(1, &_TEX[i]);
                }
                if (_texture[i]) {
                    glDeleteTextures(1, &_texture[i]);
                }
                // 清理VBO资源
                if (_VBO[i] && _cudaRsc[i]) {
                    cleanupVBO(_VBO[i], _cudaRsc[i]);
                }
                if (_VBOBox[i] && _cudaRscBox[i]) {
                    cleanupVBO(_VBOBox[i], _cudaRscBox[i]);
                }
                if (_TBO[i]) {
                    glDeleteBuffers(1, &_TBO[i]);
                }
                if (_obj[i]) {
                    delete _obj[i];
                }
            }

            if (_ourShader) {
                delete _ourShader;
            }
            // Cleanup ImGui
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();

            // glfw: terminate, clearing all previously allocated GLFW resources.
            // ------------------------------------------------------------------
            glfwTerminate();
        }
    }

    // Initialize the application
    bool init() {
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
        _window = glfwCreateWindow(_scrWidth, _scrHeight, "Interfere Detection", NULL, NULL);
        if (_window == NULL) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(_window);

        // Set user pointer to this Application instance for callbacks
        glfwSetWindowUserPointer(_window, this);

        // Set callbacks
        glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback_wrapper);
        glfwSetCursorPosCallback(_window, mouse_callback_wrapper);
        glfwSetScrollCallback(_window, scroll_callback_wrapper);
        glfwSetWindowRefreshCallback(_window, window_refresh_callback_wrapper);

        // tell GLFW to capture our mouse
        glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        // glad: load all OpenGL function pointers
        // ---------------------------------------
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return false;
        }

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(_window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        // configure global opengl state
        // -----------------------------
        glEnable(GL_DEPTH_TEST);

        // build and compile our shader zprogram
        // ------------------------------------
        _ourShader = new Shader("shader.vs", "shader.fs");

        // load and create a texture 
        // -------------------------
        // texture 1
        // ---------
        glGenTextures(1, &_texture[0]);
        glBindTexture(GL_TEXTURE_2D, _texture[0]);
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
        glGenTextures(1, &_texture[1]);
        glBindTexture(GL_TEXTURE_2D, _texture[1]);
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
        _ourShader->use();
        _ourShader->setInt("texture1", 0);
        _ourShader->setInt("texture2", 1);

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
        _maxThreadsPerBlock = prop.maxThreadsPerBlock;

        _obj[0] = new ObjLoader("object1_nut.stl");
        _obj[1] = new ObjLoader("object2_wrench.stl");

        for (int i = 0; i < 2; ++i) {
            _triNum[i] = _obj[i]->_tris.size();
            initVBOfloat3(_VAO[i], _VBO[i], _triNum[i] * 3, _cudaRsc[i]);
            initVBOfloat3(_VAOBox[i], _VBOBox[i], _triNum[i] * 48, _cudaRscBox[i]);
            initTBO(_TBO[i], _TEX[i], _triNum[i], _cudaRscHit[i]);
        }

        return true;
    }

    // Render function that can be called from both main loop and refresh callback
    void render() {
        float3 *pVertTri[2]{nullptr, nullptr}, *pVertBox[2]{nullptr, nullptr};
        int *pHit[2]{nullptr, nullptr};
        size_t sizeTri[2]{0, 0}, sizeBox[2]{0, 0}, sizeHit[2]{0, 0};
        
        // Map buffer object for writing from CUDA
        for (int i = 0; i < 2; ++i) {
            cudaGraphicsMapResources(1, &_cudaRsc[i], 0);
            cudaGraphicsMapResources(1, &_cudaRscBox[i], 0);
            cudaGraphicsMapResources(1, &_cudaRscHit[i], 0);
            cudaGraphicsResourceGetMappedPointer((void**)&pVertTri[i], &sizeTri[i], _cudaRsc[i]);
            cudaGraphicsResourceGetMappedPointer((void**)&pVertBox[i], &sizeBox[i], _cudaRscBox[i]);
            cudaGraphicsResourceGetMappedPointer((void**)&pHit[i], &sizeHit[i], _cudaRscHit[i]);
        }
        
        float time = glfwGetTime();
        //float time = 0;

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        CudaBVH* arrBVH[2]{nullptr, nullptr};
        arrBVH[0] = new CudaBVH(&_obj[0]->_tris, &_obj[0]->_box, _triNum[0], _maxThreadsPerBlock, glm::value_ptr(model), pVertTri[0], pVertBox[0]);

        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(_modelParams.translateX, _modelParams.translateY, _modelParams.translateZ));
        model = glm::rotate(model, _modelParams.rotateAngle+cos(time/2)/20, glm::vec3(_modelParams.rotateX, _modelParams.rotateY, _modelParams.rotateZ));
        model = glm::translate(model, glm::vec3(-44.31f, -4.7f, 0.0f));
        model = glm::scale(model, glm::vec3(_modelParams.scale, _modelParams.scale, _modelParams.scale));
        arrBVH[1] = new CudaBVH(&_obj[1]->_tris, &_obj[1]->_box, _triNum[1], _maxThreadsPerBlock, glm::value_ptr(model), pVertTri[1], pVertBox[1]);
        
        arrBVH[0]->boxIntersect(arrBVH[1]->_SampleSize, arrBVH[1]->d_BBoxLeaf, arrBVH[1]->d_mesh, pHit[1], pHit[0]);

#ifndef DEBUG
        static double timeCost[2][5][2] = {
            {{0.0, 0.0},{0.0, 0.0},{0.0, 0.0},{0.0, 0.0},{0.0, 0.0}},
            {{0.0, 0.0},{0.0, 0.0},{0.0, 0.0},{0.0, 0.0},{0.0, 0.0}}};
        static double avgTime[2][5] = {{0.0, 0.0, 0.0, 0.0, 0.0},{0.0, 0.0, 0.0, 0.0, 0.0}};
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 5; ++j) {
				timeCost[i][j][0] += arrBVH[i]->_TimeCost[j][0];
				timeCost[i][j][1] += arrBVH[i]->_TimeCost[j][1];
				if (timeCost[i][j][0] > 250) {
					avgTime[i][j] = timeCost[i][j][0] / timeCost[i][j][1];
					timeCost[i][j][0] = 0.0;
					timeCost[i][j][1] = 0.0;
				}
			}
		}

        static bool firstPrint = true;
        if (firstPrint) {
            std::cout << "============================================================================================" << std::endl;
            std::cout << "                                 Performance Timing Results (ms)                            " << std::endl;
            std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
            std::cout << "Object| GenerateMorton |    Radixsort   |GenerateHierachy|   InternalBox  |NodeTreeIntersect" << std::endl;
            std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
            firstPrint = false;
        }
        for (int i = 0; i < 2; ++i) {
            printf("   %d  |    %8.4f    |    %8.4f    |    %8.4f    |    %8.4f    |    %8.4f     ",
                i, avgTime[i][0], avgTime[i][1], avgTime[i][2], avgTime[i][3], avgTime[i][4]);
            std::cout << std::endl;
		}

        std::cout << "\033[2A";  // Move cursor up 2 lines
        std::cout.flush();
#endif

        for (int i = 0; i < 2; ++i) {
            cudaGraphicsUnmapResources(1, &_cudaRsc[i], 0);
            cudaGraphicsUnmapResources(1, &_cudaRscBox[i], 0);
            cudaGraphicsUnmapResources(1, &_cudaRscHit[i], 0);
		}

        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        _deltaTime = currentFrame - _lastFrame;
        _lastFrame = currentFrame;

        // input
        // -----
        processInput(_window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _texture[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, _texture[1]);

        // activate shader
        _ourShader->use();

        // pass projection matrix to shader (note that in this case it could change every frame)
        glm::mat4 projection = glm::perspective(glm::radians(_fov), (float)_scrWidth / (float)_scrHeight, 0.1f, 100.0f);
        _ourShader->setMat4("projection", projection);

        // camera/view transformation
        glm::mat4 view = glm::lookAt(_cameraPos, _cameraPos + _cameraFront, _cameraUp);
        _ourShader->setMat4("view", view);
        
        // 渲染物体
        for (int i = 0; i < 2; ++i) {
            glActiveTexture(GL_TEXTURE2 + i);
            glBindTexture(GL_TEXTURE_BUFFER, _TEX[i]);
            _ourShader->setInt("triHitBuffer", 2 + i);

            glBindVertexArray(_VAO[i]);
            // 计算三角形数量并绘制
            int triangleCount = arrBVH[i]->_SampleSize;
            _ourShader->setFloat4("color", _colorObj[0], _colorObj[1], _colorObj[2], _colorObj[3]);
            _ourShader->setFloat4("colorHit", _colorHit[0], _colorHit[1], _colorHit[2], _colorHit[3]);
            _ourShader->setInt("nbVertexPerElement", 3);
            glDrawArrays(GL_TRIANGLES, 0, triangleCount * 3);

            glBindVertexArray(_VAOBox[i]);
            _ourShader->setFloat4("color", _colorBox[0], _colorBox[1], _colorBox[2], _colorBox[3]);
            _ourShader->setFloat4("colorHit", _colorHit[0], _colorHit[1], _colorHit[2], _colorHit[3]);
            _ourShader->setInt("nbVertexPerElement", 24);
            glDrawArrays(GL_LINES, 0, triangleCount * 24);
        }

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create ImGui control panel
        ImGui::Begin("Model Matrix Controls");
        
        // Display FPS
        float fps = 1.0f / _deltaTime;
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Frame Time: %.3f ms", _deltaTime * 1000.0f);
        ImGui::Separator();
        
        ImGui::Text("Adjust Object 2 (Wrench) Transform");
        ImGui::Separator();
        
        ImGui::Text("Translation");
        ImGui::SliderFloat("Translate X", &_modelParams.translateX, -15.381198f, 24.618802f);
        ImGui::SliderFloat("Translate Y", &_modelParams.translateY, -16.0f, 24.0f);
        ImGui::SliderFloat("Translate Z", &_modelParams.translateZ, -20.0f, 20.0f);
        
        ImGui::Separator();
        ImGui::Text("Rotation");
        ImGui::SliderFloat("Rotate Angle", &_modelParams.rotateAngle, -4.9816f, 1.3016f);
        ImGui::SliderFloat("Rotate Axis X", &_modelParams.rotateX, -1.0f, 1.0f);
        ImGui::SliderFloat("Rotate Axis Y", &_modelParams.rotateY, -1.0f, 1.0f);
        ImGui::SliderFloat("Rotate Axis Z", &_modelParams.rotateZ, -1.0f, 1.0f);
        
        ImGui::Separator();
        ImGui::Text("Scale");
        ImGui::SliderFloat("scale", &_modelParams.scale, 0.5f, 3.0f);
        
        ImGui::Separator();
        if (ImGui::Button("Reset to Default")) {
            _modelParams.translateX = 4.618802f;
            _modelParams.translateY = 4.0f;
            _modelParams.translateZ = 0.0f;
            _modelParams.rotateAngle = -1.84f;
            _modelParams.rotateX = 0.0f;
            _modelParams.rotateY = 0.0f;
            _modelParams.rotateZ = 1.0f;
            _modelParams.scale = 1.75f;
        }
        
        ImGui::Separator();
        ImGui::Text("Press F1 to toggle mouse capture");
        ImGui::Text("Current: %s", _mouseCaptured ? "Camera Control" : "UI Interaction");
        
        ImGui::End();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        delete (arrBVH[0]);
        delete (arrBVH[1]);
    }

    // Main run loop
    void run() {
        // render loop
        // -----------
        while (!glfwWindowShouldClose(_window)) {
            render();
            
            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
            glfwSwapBuffers(_window);
            glfwPollEvents();
        }

        std::cout << std::endl;
    }
};

int main(int argc, char* argv[]) {
    //freopen("D:\\File\\Graduation_Project\\interfereDetection\\debug.log", "w", stdout);
    Application app;
    if (app.init()) {
        app.run();
    }
	checkCudaErrors(cudaDeviceReset());
	return 0;
}