# Interfere Detection

CUDA 加速的 3D 模型实时干涉检测系统，基于 GPU 并行计算实现两个三角网格模型之间的碰撞检测与可视化。

## 演示

![demo](demo.gif)

## 技术栈

- **CUDA 12.9** — GPU 并行计算
- **C++17** — 核心逻辑
- **OpenGL 3.3** — 实时渲染
- **GLFW / GLM / Dear ImGui** — 窗口、数学、UI
- **CUB** — GPU 基数排序

## 核心算法

1. **Morton 码生成** — 三角形空间编码
2. **CUB 基数排序** — GPU 并行排序
3. **BVH 层次树构建** — 空间加速结构
4. **Guigue-Devillers 三角形相交检测** — 学术界经典算法，EPSILON=1e-12 数值稳定性处理

## 性能

| 阶段 | Object 0 (≈6,000△) | Object 1 (≈12,000△) |
|------|---------------------|----------------------|
| Morton 码生成 | 0.64ms | 0.57ms |
| 基数排序 | 1.26ms | 0.99ms |
| BVH 层次树 | ~0ms | ~0ms |
| 包围盒计算 | 1.26ms | 2.65ms |
| 碰撞检测 | 1.12ms | ~0ms |

两个模型共约 18,000 三角形，全流程 **≤15ms/帧**（60fps）。

## 构建

**环境要求**：
- Windows 10/11 x64
- Visual Studio 2022 (v143)
- CUDA Toolkit 12.9
- GLFW 3.x / GLM 1.0.2+ / Glad

```bash
msbuild interfereDetection.sln /p:Configuration=Release /p:Platform=x64
```

## 项目结构

```
├── kernel.cu          # 主程序入口 + Application 类
├── CudaBVH.cu/.cuh    # BVH 树构建与遍历
├── Triangle.cu/.cuh   # 三角形几何与相交算法
├── BBox.cuh           # 轴对齐包围盒
├── Point.cu/.cuh      # 3D 点/向量
├── objloader.cpp/.h   # STL 模型加载
├── shader.vs / shader.fs  # OpenGL 着色器
├── Shader.h           # 着色器管理
└── imgui/             # Dear ImGui 集成
```
