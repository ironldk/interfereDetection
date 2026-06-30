# AGENTS.md

This repository is a CUDA-accelerated 3D interference detection system built with Visual Studio.

## Build Commands

**Platform**: Windows x64 only
**Toolchain**: Visual Studio 2022 (v143 toolset) with CUDA 12.9
**C++ Standard**: C++17
**Compute Capability**: 6.1 (sm_61)

### Building
```bash
# Open in Visual Studio 2022 and build:
interfereDetection.sln

# Or use MSBuild command line:
msbuild interfereDetection.sln /p:Configuration=Release /p:Platform=x64
msbuild interfereDetection.sln /p:Configuration=Debug /p:Platform=x64
```

### Running Tests
No automated test framework is present. Run the executable manually:
```
x64/Debug/interfereDetection.exe
x64/Release/interfereDetection.exe
```

## Project Structure

**Entry Point**: `kernel.cu` - Contains `Application` class and main() function
**Core Components**:
- `Triangle.cu/.cuh` - Triangle geometry and intersection algorithms
- `BBox.cuh` - Axis-aligned bounding box operations (header-only)
- `CudaBVH.cu/.cuh` - BVH tree construction and traversal
- `Point.cu/.cuh` - 3D point/vector class
- `objloader.cpp/.h` - STL file loader
- `Shader.h` - OpenGL shader management
- `Node.cuh` - Base node class for BVH tree
- `LeafNode.cuh` - Leaf node for BVH tree
- `InternalNode.cuh` - Internal node for BVH tree
- `stb_image.h` - Image loading

**ImGui Integration** (imgui/ directory):
- Dear ImGui v1.91.0 library
- ImGui GLFW and OpenGL3 backends
- Added for runtime parameter adjustment

**Models** (loaded in kernel.cu):
- `object1_nut.stl` - First 3D model (nut)
- `object2_wrench.stl` - Second 3D model (wrench)

## Code Style Guidelines

### File Naming
- CUDA source: `.cu` extension
- CUDA headers: `.cuh` extension (not `.h`)
- C++ source: `.cpp` extension
- C++ headers: `.h` extension
- Shaders: `.vs` (vertex), `.fs` (fragment)

### Imports and Includes
- Use local includes with `#include "filename.cuh"` for CUDA headers
- Use system includes with angle brackets: `#include <cuda_runtime.h>`
- CUDA includes must precede standard headers when mixing
- Place includes in order: CUDA headers → standard library headers → local headers

### Naming Conventions
- **Classes**: PascalCase (`Triangle`, `BBox`, `CudaBVH`)
- **Methods/Functions**: PascalCase (`Intersect`, `Transform`, `GenerateBBox`)
- **Variables**: camelCase (`cameraPos`, `_SampleSize`)
- **Private members**: Underscore prefix (`_min`, `_max`, `_SampleSize`)
- **Parameters**: Use prefixes to indicate direction:
  - `i` prefix for input (`iSampleSize`, `iMatrixModel`)
  - `o` prefix for output (`oBBoxLeaf`, `oMCode`)
- **Constants**: UPPER_SNAKE_CASE (`SCR_WIDTH`, `EPSILON`)
- **Macros**: UPPER_SNAKE_CASE with `#define`

### Formatting
- **Indentation**: Tabs (display as 4 spaces)
- **Braces**: Opening brace on same line for methods, new line for control flow
- **Spacing**: Space after keywords, before/after operators
- **Line length**: No strict limit, but keep code readable

```cpp
// Method example
__host__ __device__ bool Intersect(const Triangle& r) const {
    if (condition) {
        return false;
    }
    return true;
}
```

### CUDA-Specific Guidelines
- Use `__host__ __device__` dual qualifier for functions used in both host and device code
- Use `inline` for small, frequently called device functions
- Kernel parameters: use `i`/`o` prefixes for input/output clarity
- Use `new(&location) Type()` for placement new in device code (BBox construction)
- Always check bounds: `if (idx >= iSampleSize) return;`

### Memory Management
- Host memory: `cudaMalloc()` → `cudaFree()`
- Device pointers prefix with `d_` (`d_mesh`, `d_BBoxLeaf`, `d_internalNodes`)
- Use CUDA-OpenGL interop: `cudaGraphicsGLRegisterBuffer()`, `cudaGraphicsMapResources()`
- Always check CUDA errors: `cudaError_t`, `checkCudaErrors()`, `CubDebugExit()`
- Use `cudaDeviceSynchronize()` after kernel launches when timing or debugging

### Error Handling
- Use `assert(cudaStatus == cudaSuccess)` for CUDA operations
- Use `checkCudaErrors()` macro from helper_cuda.h
- Print errors with `std::cout` (not cerr)
- Use `CUDA_DEBUG_PRINT` macro for conditional debug output (defined in DEBUG mode)
- Check OpenGL shader compilation: `glGetShaderiv(shader, GL_COMPILE_STATUS, &success)`

### Comments
- Use English comments for technical explanations
- Use Chinese comments sparingly for domain-specific notes (existing pattern)
- Comment complex algorithms (Morton codes, triangle intersection)
- Add inline comments for mathematical operations

### Mathematical/Geometry Code
- Use `EPSILON` for floating-point comparisons (`1e-12`)
- Use `fabsf()` for absolute value in device code
- Vector operations: overload operators in `Point` class (`+`, `-`, `*`, `/`)
- Use GLM matrices for transformations (`glm::mat4`, `glm::translate`)
- Triangle intersection: Guigue and Devillers algorithm (see `Triangle::Intersect`)

## Memory Layout and CUDA-OpenGL Interop

### Critical: Memory Alignment Guarantees
All classes used in CUDA-OpenGL interop have strict memory layout guarantees:

**Point Class**:
- `#pragma pack(push, 1)` - 1-byte alignment, no padding
- `static_assert(sizeof(Point) == 12)` - Exactly 3 floats
- Memory layout: `[x(float)][y(float)][z(float)]` (12 bytes total)

**Triangle Class**:
- `#pragma pack(push, 1)`
- `static_assert(sizeof(Triangle) == 36)` - Exactly 3 Points
- Memory layout: `[Point a][Point b][Point c]` (36 bytes total)

**BBox Class**:
- `#pragma pack(push, 1)`
- `static_assert(sizeof(BBox) == 288)` - Exactly 24 Points (for wireframe rendering)
- Memory layout: 24 consecutive Point objects (288 bytes total)

### Why This Matters
- OpenGL VBOs store raw float3 data
- CUDA code `reinterpret_cast`'s these buffers to class pointers
- Without `#pragma pack`, compilers may add padding for alignment
- Static assertions ensure compile-time validation of memory layout

### Memory Layout Guarantee
All geometry classes must pass `sizeof` checks at compile time. If you add or modify fields in `Point`, `Triangle`, or `BBox`, update the corresponding `static_assert`.

## External Dependencies

**Required Libraries**:
- CUDA Toolkit 12.9
- GLFW 3.x
- OpenGL (via Glad)
- GLM 1.0.2+ (mathematics library)
- CUB (CUDA Unbound) - included via headers
- Dear ImGui v1.91.0 - included in imgui/ directory

**Include Paths** (from .vcxproj):
- `D:\Projects\glfw\include`
- `D:\Projects\glm-1.0.2`
- `D:\Program Files\glad\include`
- `C:\ProgramData\NVIDIA Corporation\cuda-samples\Common`

**Linked Libraries**:
- `glfw3.lib`, `opengl32.lib`, `cudart_static.lib`, `kernel32.lib`, `user32.lib`, `gdi32.lib`, `winspool.lib`, `comdlg32.lib`, `advapi32.lib`, `shell32.lib`, `ole32.lib`, `oleaut32.lib`, `uuid.lib`, `odbc32.lib`, `odbccp32.lib`

## Key Algorithms

**BVH Construction**:
1. Generate Morton codes from triangle centers
2. Radix sort using CUB
3. Build hierarchical tree structure
4. Compute bounding boxes bottom-up

**Triangle Intersection**:
- Uses Guigue and Devillers method
- Handles coplanar and non-coplanar cases
- EPSILON-based tolerance for numerical stability (always enabled)

## Rendering Pipeline

- OpenGL 3.3 Core Profile
- Wireframe rendering: `glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)`
- Transparency: `glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)`
- Two render objects with CUDA-OpenGL interop buffers
- Camera: WASD movement, mouse look, scroll zoom
- Collision detection result via texture buffer (`triHitBuffer`)
- ImGui UI for runtime parameter adjustment
- **Dynamic window resizing**: Proper viewport and projection matrix updates on window size change
- **Continuous refresh during resize/move**: Window refresh callback ensures continuous rendering while dragging or resizing

### Collision Detection Rendering
The shader uses a texture buffer to visualize collision results:
- `triHitBuffer`: `isamplerBuffer` containing collision flags per triangle
- `color`: Base color for non-colliding triangles
- `colorHit`: Highlight color for colliding triangles

## Performance Timing System

### Time Cost Measurement
The system measures performance across 5 stages:
- **Time 0**: Generate Morton codes and bounding boxes
- **Time 1**: Radix sort
- **Time 2**: Generate BVH hierarchy
- **Time 3**: Assign bounding boxes to internal nodes
- **Time 4**: Intersection detection

### Console Output
Performance timing results are displayed in the console with:
- Fixed header text (only printed once)
- Separate rows for each object (Nut and Wrench)
- Refreshing numerical values (using `\r` carriage return and cursor movement)
- 5 timing metrics displayed in aligned columns
- Values formatted to 2 decimal places (ms)

## ImGui UI Features

### Runtime Model Matrix Controls
- **Toggle Mouse Capture**: Press F1 key to switch between camera control and UI interaction modes
- **Translation Sliders**: Adjust X, Y, Z position of Object 2 (Wrench)
- **Rotation Controls**: Adjust rotation angle and axis (X, Y, Z components)
- **Scale Sliders**: Uniform scaling of Object 2
- **Reset Button**: Quick reset to default values
- **FPS Display**: Real-time FPS and frame time metrics at top of panel

## Shader Details

### Vertex Shader (shader.vs)
```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
flat out int triID;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int nbVertexPerElement;

void main() {
    // model matrix disabled - using transformed vertices from CUDA
    gl_Position = projection * view * vec4(aPos, 1.0f);
    triID = gl_VertexID / nbVertexPerElement;
}
```

### Fragment Shader (shader.fs)
```glsl
#version 330 core
out vec4 FragColor;
flat in int triID;

uniform vec4 color;
uniform vec4 colorHit;
uniform isamplerBuffer triHitBuffer;

void main() {
    int hit = texelFetch(triHitBuffer, triID).r;
    FragColor = hit > 0 ? colorHit : color;
}
```

## Common Patterns

### Kernel Launch Pattern
```cpp
int blocks = (iSampleSize + _ThreadsPerBlock - 1) / _ThreadsPerBlock;
KernelName<<<blocks, _ThreadsPerBlock>>>(params...);
cudaDeviceSynchronize();
```

### CUDA Timing Pattern
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// ... code ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### Resource Cleanup Pattern
```cpp
// Always cleanup in reverse order
cudaFree(d_ptr);
d_ptr = nullptr;
// For CUDA-OpenGL resources
cudaGraphicsUnregisterResource(cudaRsc);
glDeleteBuffers(1, &vbo);
```

### Console Refresh Pattern
```cpp
static bool firstPrint = true;
if (firstPrint) {
    // Print header text only once
    std::cout << "Header text..." << std::endl;
    firstPrint = false;
}

// Use \r to return to start of line for refreshing values
std::cout << "\r";
printf("Formatted values...");
std::cout.flush();
```

## Application Class Architecture

The code is organized into a single `Application` class in `kernel.cu`:

**Key Features**:
- Eliminates all global variables by encapsulating state within the class
- Uses `glfwSetWindowUserPointer()` to pass class instance to GLFW callbacks
- Static callback wrappers retrieve the Application instance and delegate to member methods
- Clean separation of concerns: `init()`, `render()`, `run()`, and `cleanup()` methods

**Private Members**:
- Window, camera, and rendering state variables (underscore prefix `_`)
- Static callback wrapper functions
- Callback implementations as private methods
- Helper functions (`initVBOfloat3`, `initTBO`, `cleanupVBO`, `processInput`)

**Public Interface**:
- `Application()` - Default constructor
- `~Application()` - Destructor that calls `cleanup()`
- `init()` - Initializes GLFW, OpenGL, ImGui, and loads resources
- `render()` - Performs a single frame render
- `run()` - Main render loop
- `cleanup()` - Releases all resources

## Notes

- Project uses Chinese comments in some places (existing codebase pattern)
- Debug builds have extensive CUDA_DEBUG_PRINT output
- Model files are STL format (3D printing format, not Wavefront OBJ)
- Default window size: 2250x1000 (configurable in Application class)
- Camera initial position: (4.618802, 4.0, 24.0)
- Vertices are transformed on GPU before rendering (model matrix applied in CUDA kernel)
- Press F1 key to toggle between camera control and UI interaction modes
- TAB key is used for ImGui widget navigation in UI interaction mode
- Time cost measurements are per-object to avoid data races

## Common Patterns (Continued)

### GLFW Callback with User Pointer Pattern
```cpp
// In init():
glfwSetWindowUserPointer(_window, this);
glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback_wrapper);

// Static wrapper:
static void framebuffer_size_callback_wrapper(GLFWwindow* window, int width, int height) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    app->framebufferSizeCallback(window, width, height);
}

// Member implementation:
void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    _scrWidth = width;
    _scrHeight = height;
    glViewport(0, 0, width, height);
}
```
