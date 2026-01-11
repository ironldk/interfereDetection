#ifndef CUDABVH_CUH_
#define CUDABVH_CUH_

#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <intrin.h>
#include <chrono>
#include <Windows.h>
#include "BBox.cuh"
#include "Node.cuh"
#include "LeafNode.cuh"
#include "InternalNode.cuh"
#include "BVHTree.cuh"
#include "objloader.h"
#include <vector>
#include <glm/glm.hpp>

// OpenGL includes for CUDA-OpenGL interop
#include <cuda_gl_interop.h>

#define HASH_64 1
#if HASH_64
    typedef unsigned __int64 HashType;
#else
    typedef unsigned int HashType;
#endif

//#define DEBUG
#ifdef DEBUG
#define CUDA_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define CUDA_DEBUG_PRINT(...)
#endif

__global__ void GenerateBBox(int iSampleSize, Triangle* iTriangle, BBox* oBBoxLeaf, float iMatrixModel[16]);

__global__ void Morton3DCuda(int iSampleSize, const BBox* iBBoxs,
	HashType* oMCode, int* oObjectIDs);

__device__ HashType ExpandBits(HashType iv);

__global__ void AssignLeafNodes(int iSampleSize, LeafNode* ipLeafNodes,
    InternalNode* ipInternalNodes, int* iSortedObjectIDs, BBox* iBBoxs);

__global__ void AssignInternalNodes(int iSampleSize, LeafNode* ipLeafNodes,
    InternalNode* ipInternalNodes, HashType* iSortedMortons);

__device__ int2 DetermineRange(HashType* iSortedMortons, int iNumObjects, int idx);

__device__ int Sign(int ix);

__device__ int CommonLeadingBits(HashType* iSortedMortons, int ix, int iy, int iNumObjects);

__device__ int FindSplit(HashType* iSortedMortons, int iFirst, int iLast);

__global__ void ValuesKernel(int SAMPLE_SIZE, int* keys);
__global__ void InternalNodeBBox(int iSampleSize, LeafNode* iLeafNodes, InternalNode* iInternalNodes, BBox* iBBoxs, int* iReadyFlags);
__global__ void CompleteBBox(int iSampleSize, BBox* iBBoxLeaf, BBox* iBBoxInner);
__host__ __device__ bool IntersectRayAABB(const float3& start,
	const float3& dir, const float3& bmin, const float3& bmax, float& t);
__host__ __device__ inline float3 cross(const float3& a, const float3& b);
__host__ __device__ inline float dot(const float3& a, const float3& b);
inline __host__ __device__ float3 normalize(const float3& v);
__device__ __host__ float3 f3_add(float3 A, float3 B);
__device__ __host__ float3 f3_sub(float3 A, float3 B);
__device__ __host__ float f3_dot(float3 A, float3 B);
__device__ __host__ float3 f3_crss(float3 A, float3 B);
__device__ __host__ float3 f3_sclrmult(float val, float3 A);
// Moller and Trumbore's method
__host__ __device__ bool IntersectRayTriTwoSided(
	const float3& p, const float3& dir,
	const float3& a, const float3& b, const float3& c,
	float& t, float& u, float& v);
__global__ void intersectKernel(int iSampleSize2, BBox* iBBoxLeaf2, Triangle* iTriangle2,
    int* oHit2, int* oHit, int iSampleSize, InternalNode* iInternalNodes, LeafNode* leafNodes, Triangle* myMesh);

class CudaBVH{
public:
    CudaBVH(std::vector<Triangle>* iTris, int sample_size,
        int threads_per_block, float iMatrixModel[16], float3* oGL_d_mesh, float3* oGL_d_box);
    ~CudaBVH();

    void generateValues(int* keys);

	int _SampleSize;
	int _ThreadsPerBlock;

    std::vector<Triangle> *h_myMesh = nullptr;
    Triangle *d_mesh = nullptr;
    BBox *d_BBoxLeaf = nullptr;
    BBox *d_BBoxInner = nullptr;
    LeafNode* d_leafNodes = nullptr;
    InternalNode* d_internalNodes = nullptr;

    void boxIntersect(int iSampleSize2, BBox* iBBoxLeaf2, Triangle* iTriangle2, int* oHit2, int* oHit);
 //bool intersect(const float3& ray_orig, const float3& ray_dir,
 //	float& outT, float& outU, float& outV, int& outIdx);
};

#endif