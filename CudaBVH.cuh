#ifndef CUDABVH_CUH_
#define CUDABVH_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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
#include <cub/util_allocator.cuh>
#include "test/test_util.h"
#include <intrin.h>
#include <chrono>
#include <Windows.h>
#include "BBox.cuh"
#include "Node.cuh"
#include "LeafNode.cuh"
#include "InternalNode.cuh"
#include "BVHTree.cuh"
#include "objloader.h"
//#include<helper_math.h>
using namespace std;
using namespace cub;
CachingDeviceAllocator g_allocator(true);
#define HASH_64 1
#define EPSILON 0.000000000001
#if HASH_64
    typedef unsigned __int64 HashType;
#else
    typedef unsigned int HashType;
#endif

__device__ int findSplit(HashType* sortedMortonCodes, int first, int last);
__device__ int delta(HashType* sortedMortonCodes, int x, int y, int numObjects);
__device__ int sign(int x);
__device__ int2 determineRange(HashType* sortedMortonCodes, int numObjects, int idx);
__device__ HashType expandBits(HashType v);
__global__ void assignInternalNodes(int SAMPLE_SIZE, HashType* sortedMortonCodes,
	LeafNode* leafNodes, InternalNode* internalNodes, int* sortedObjectIDs);

#if 0
__global__ void morton3DCuda(int SAMPLE_SIZE, HashType* c, const BBox* objects);
#else
class MortonRec {
public:
	__device__ MortonRec(int sample_size, const BBox& bbox, int idx);
    float x, y, z;
    float xx, yy, zz;
    HashType ex, ey, ez;
    HashType m;
};
__global__ void morton3DCuda(int SAMPLE_SIZE, HashType* c, const BBox* objects,
	MortonRec* mor);
#endif

__global__ void valuesKernel(int SAMPLE_SIZE, int* keys);
__global__ void internalNodeBBox(int SAMPLE_SIZE, int* atom,
    InternalNode* internalNodes, LeafNode* leafNodes, BBox* d_myBBox);
__global__ void assignLeafNodes
    (int SAMPLE_SIZE, LeafNode* leafNodes, int* sortedObjectIDs, BBox* bbox);
__host__ __device__ bool IntersectRayAABB(const float3& start,
	const float3& dir, const float3& bmin, const float3& bmax, float& t);
__host__ __device__ bool IntersectAABBAABB(const BBox& b, const BBox& a);
__host__ __device__ inline float3 cross(const float3& a, const float3& b);
__host__ __device__ inline float dot(const float3& a, const float3& b);
inline __host__ __device__ float3 normalize(const float3& v);
__device__ __host__ float3 f3_add(float3 A, float3 B);
__device__ __host__ float3 f3_sub(float3 A, float3 B);
__device__ __host__ float f3_dot(float3 A, float3 B);
__device__ __host__ float3 f3_crss(float3 A, float3 B);
__device__ __host__ float3 f3_sclrmult(float val, float3 A);
__device__ __host__ float f_clamp(float n, float min, float max);
// Moller and Trumbore's method
__host__ __device__ bool IntersectRayTriTwoSided(
	const float3& p, const float3& dir,
	const float3& a, const float3& b, const float3& c,
	float& t, float& u, float& v);
__device__ __host__ float edge_to_edge(float3 p1, float3 q1, float3 p2, float3 q2,
    float& s, float& t, float3& c1, float3& c2);
__device__ __host__ float point_to_triangle
    (float3 pt, float3& ptt, float3 a, float3 b, float3 c);
//, float3 &pr1, float3 &pr2) pr1,pr2 is nearest pair of points
__device__ __host__ bool IntersectTriangleTriangle(
    const float3 a1, const float3 b1, const float3 c1,
    const float3 a2, const float3 b2, const float3 c2);
__device__ bool intersect(const BBox& bbox2, const Triangle& tri2, int* outHit,
    InternalNode* internalNodes, LeafNode* leafNodes, Triangle* myMesh);
__global__ void intersectKernel(
    int total_boxes2,
    BBox* bboxes2, Triangle* tris2, int* outHit, int* outHit2,
    InternalNode* internalNodes, LeafNode* leafNodes, Triangle* myMesh);

class CudaBVH{
public:
	int SAMPLE_SIZE;
	int THREADS_PER_BLOCK;
	BVHTree myTree, d_myTree;
    vector<Triangle> myMesh;
    vector<BBox> myBBox;
    Triangle *d_myMesh = nullptr;
    BBox *d_myBBox = nullptr;
    MortonRec *mor = nullptr;
    void generateValues(int* keys);
    BVHTree generateBVHTree(int* values, // element index in the unsorted array
        BBox* objects, Triangle* tris);
public:
    ~CudaBVH();
// 	CudaBVH();
// 	CudaBVH(int sample_size, int threads_per_block);
    CudaBVH(BBox* objects, Triangle* tris, int sample_size, int threads_per_block);
    void Init(BBox* objects, Triangle* tris, int sample_size, int threads_per_block);
    void generateSampleDataset(BBox* objects);
#if 0
    void printBVH(int idx, int level);
#else
    void printBVH(InternalNode* internalNodes, LeafNode* leafNodes);
#endif

#if 0
    void drawBVHRecursive(int idx, int level);
    void draw();
#else
    void draw(int levelDisplay);
#endif
    void drawTriangles();
    void drawTrianglesDEBUG();
    void boxIntersect(int boxes2_count, BBox* bboxes2, Triangle* tris2,
		int* outHit, int* outHit2);
	//bool intersect(const float3& ray_orig, const float3& ray_dir,
	//	float& outT, float& outU, float& outV, int& outIdx);
};

#endif