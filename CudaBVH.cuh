#ifndef CUDABVH_CUH_
#define CUDABVH_CUH_

#include <helper_cuda.h>
#include <stdlib.h>
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
#include "objloader.h"
#include <vector>
#include <glm/glm.hpp>

// OpenGL includes for CUDA-OpenGL interop
#include <cuda_gl_interop.h>


__global__ void Morton3DCuda(int iSampleSize, Triangle* iTriangle, BBox* oBBoxLeaf,
    float iMatrixModel[16], HashType* oMCode, int* oObjectIDs, float* imin);

__device__ HashType ExpandBits(HashType iv);

__global__ void AssignLeafNodes(int iSampleSize, LeafNode* ipLeafNodes,
    InternalNode* ipInternalNodes, int* iSortedObjectIDs, BBox* iBBoxs);

__global__ void AssignInternalNodes(int iSampleSize, LeafNode* ipLeafNodes,
    InternalNode* ipInternalNodes, HashType* iSortedMortons);

__device__ int2 DetermineRange(HashType* iSortedMortons, int iNumObjects, int idx);

__device__ int Sign(int ix);

__device__ int CommonLeadingBits(HashType* iSortedMortons, int ix, int iy, int iNumObjects);

__device__ int FindSplit(HashType* iSortedMortons, int iFirst, int iLast);

__global__ void InternalNodeBBox(int iSampleSize, LeafNode* iLeafNodes, InternalNode* iInternalNodes, BBox* iBBoxs, int* iReadyFlags);
__global__ void CompleteBBox(int iSampleSize, BBox* iBBoxLeaf, BBox* iBBoxInner);

__global__ void intersectKernel(int iSampleSize2, BBox* iBBoxLeaf2, Triangle* iTriangle2,
    int* oHit2, int* oHit, int iSampleSize, InternalNode* iInternalNodes, LeafNode* leafNodes, Triangle* myMesh);

class CudaBVH{
public:
    CudaBVH(std::vector<Triangle>* iTris, BBox* iBBox, int sample_size,
        int threads_per_block, float iMatrixModel[16], float3* oGL_d_mesh, float3* oGL_d_box);
    ~CudaBVH();

	int _SampleSize;
	int _ThreadsPerBlock;

    Triangle *d_mesh = nullptr;
    BBox *d_BBoxLeaf = nullptr;
    BBox *d_BBoxInner = nullptr;
    LeafNode* d_leafNodes = nullptr;
    InternalNode* d_internalNodes = nullptr;
    double _TimeCost[5][2];

    void boxIntersect(int iSampleSize2, BBox* iBBoxLeaf2, Triangle* iTriangle2, int* oHit2, int* oHit);
};

#endif