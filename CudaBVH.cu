#include "CudaBVH.cuh"
#include <cub/util_allocator.cuh>
using std::cout;
using std::endl;

cub::CachingDeviceAllocator g_allocator(true);

CudaBVH::CudaBVH(std::vector<Triangle>* iTris, int iSampleSize,
	int iThreadsPerBlock, float iMatrixModel[16], float3* oGL_d_mesh, float3* oGL_d_box)
:
	_SampleSize(iSampleSize),
	_ThreadsPerBlock(iThreadsPerBlock),
	h_myMesh(iTris),
	d_mesh(nullptr),
	d_BBoxLeaf(nullptr),
	d_BBoxInner(nullptr),
	d_leafNodes(nullptr),
	d_internalNodes(nullptr)
{
// generate morton code
	cub::DoubleBuffer<HashType> d_sortedKeys;
	cub::DoubleBuffer<int> d_sortedValues;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sortedKeys.d_buffers[0], sizeof(HashType) * _SampleSize));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sortedKeys.d_buffers[1], sizeof(HashType) * _SampleSize));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sortedValues.d_buffers[0], sizeof(int) * _SampleSize));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sortedValues.d_buffers[1], sizeof(int) * _SampleSize));


	HashType* d_objKeys = d_sortedKeys.Current();
	int* d_objValues = d_sortedValues.Current();
	d_mesh = reinterpret_cast<Triangle*>(oGL_d_mesh);
	d_BBoxLeaf = reinterpret_cast<BBox*>(oGL_d_box);
	d_BBoxInner = reinterpret_cast<BBox*>(oGL_d_box) + _SampleSize;
	float *d_matrixModel = nullptr;
	cudaMalloc((void**)&d_matrixModel, 16*sizeof(float));
	cudaMemcpy(d_mesh, &(*h_myMesh)[0], _SampleSize * sizeof(Triangle), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixModel, iMatrixModel, 16 * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	GenerateBBox<<<(_SampleSize+_ThreadsPerBlock-1)/_ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_mesh, d_BBoxLeaf, d_matrixModel);
	Morton3DCuda<<<(_SampleSize+_ThreadsPerBlock-1)/_ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_BBoxLeaf, d_objKeys, d_objValues);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	CUDA_DEBUG_PRINT("It took me %f milliseconds to generate morton codes.\n", milliseconds);

	cudaFree(d_matrixModel);
	d_matrixModel = nullptr;

// Radix sort
	// Allocate temporary storage
	size_t temp_storage_bytes = 0;
	void* d_temp_storage = nullptr;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortedKeys, d_sortedValues, _SampleSize));
	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortedKeys, d_sortedValues, _SampleSize));
	cudaEventRecord(stop);

	CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
	d_objKeys = d_sortedKeys.Current();
	d_objValues = d_sortedValues.Current();


	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	CUDA_DEBUG_PRINT("It took me %f milliseconds to run parallel radix sort.\n", milliseconds);

// Generate Hierachy
	// Construct leaf nodes.
	// Note: This step can be avoided by storing the tree in a slightly different way.
	cudaMalloc((void**)&d_leafNodes, _SampleSize * sizeof(LeafNode));
	cudaMalloc((void**)&d_internalNodes, (_SampleSize - 1) * sizeof(InternalNode));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	AssignLeafNodes<<<(_SampleSize+_ThreadsPerBlock-1)/_ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_leafNodes, d_internalNodes, d_objValues, d_BBoxLeaf);
	cudaDeviceSynchronize();
	AssignInternalNodes<<<(_SampleSize+_ThreadsPerBlock-1)/_ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_leafNodes, d_internalNodes, d_objKeys);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	CUDA_DEBUG_PRINT("It took me %f milliseconds to generate hierachy.\n", milliseconds);

	CubDebugExit(g_allocator.DeviceFree(d_sortedKeys.d_buffers[0]));
	CubDebugExit(g_allocator.DeviceFree(d_sortedKeys.d_buffers[1]));
	CubDebugExit(g_allocator.DeviceFree(d_sortedValues.d_buffers[0]));
	CubDebugExit(g_allocator.DeviceFree(d_sortedValues.d_buffers[1]));

// Assign bounding box to internal nodes
	int* d_ReadyFlags = nullptr;
	cudaMalloc((void**)&d_ReadyFlags, (_SampleSize - 1) * sizeof(int));
	cudaMemset(d_ReadyFlags, 0, (_SampleSize - 1) * sizeof(int));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	InternalNodeBBox<<<(_SampleSize+_ThreadsPerBlock-1)/_ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_leafNodes, d_internalNodes, d_BBoxInner, d_ReadyFlags);
	CompleteBBox<<<(_SampleSize+_ThreadsPerBlock-1)/_ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_BBoxLeaf, d_BBoxInner);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	CUDA_DEBUG_PRINT("It took me %f milliseconds to assign bounding box.\n", milliseconds);

	cudaFree(d_ReadyFlags);
	d_ReadyFlags = nullptr;
}

CudaBVH::~CudaBVH() {
	h_myMesh = nullptr;
	d_mesh = nullptr;
	d_BBoxLeaf = nullptr;
	d_BBoxInner = nullptr;
	cudaFree(d_leafNodes);
	cudaFree(d_internalNodes);
	d_leafNodes = nullptr;
	d_internalNodes = nullptr;
}

__global__ void GenerateBBox(int iSampleSize, Triangle* iTriangle, BBox* oBBoxLeaf, float iMatrixModel[16]) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= iSampleSize) {
		return;
	}
	iTriangle[idx].Transform(iMatrixModel);
	new(&oBBoxLeaf[idx]) BBox(iTriangle[idx]);

#ifdef DEBUG
	if (idx < 50) {
		CUDA_DEBUG_PRINT("idx=%d, &iTriangle[idx]=%p, &oBBoxLeaf[idx]=%p\n",
			idx, &iTriangle[idx], &oBBoxLeaf[idx]);
	}
#endif
}

__global__ void Morton3DCuda(int iSampleSize, const BBox* iBBoxs, HashType* oMCode, int* oObjectIDs) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < iSampleSize) {
		float x, y, z;
		iBBoxs[idx].GetCenter(x, y, z);
#if HASH_64
		float xx = x * 1024.0f * 1024.0f;
		float yy = y * 1024.0f * 1024.0f;
		float zz = z * 1024.0f * 1024.0f;
#else
		_x = _x * 1023.0f;
		_y = _y * 1023.0f;
		_z = _z * 1023.0f;
#endif
		HashType ex = ExpandBits((HashType)((double)xx));
		HashType ey = ExpandBits((HashType)((double)yy));
		HashType ez = ExpandBits((HashType)((double)zz));
		oMCode[idx] = ((ex << 2) + (ey << 1) + ez);		// 防止Morton码重复
		oObjectIDs[idx] = idx;
	}
}

#if HASH_64
// Expand a 21-bit integer into 63 bits by inserting 2 zeros before each bit.
__device__ HashType ExpandBits(HashType iv) {
	//iv0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001'1111'1111'1111'1111'1111u;

	iv = (iv *
		0b0000'0000'0000'0000'0000'0000'0000'0001'0000'0000'0000'0000'0000'0000'0000'0001u) &
		0b1111'1111'1111'1111'0000'0000'0000'0000'0000'0000'0000'0000'1111'1111'1111'1111u;
	//iv0b0000'0000'0001'1111'0000'0000'0000'0000'0000'0000'0000'0000'1111'1111'1111'1111u;

	iv = (iv *
		0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001'0000'0000'0000'0001u) &
		0b0000'0000'1111'1111'0000'0000'0000'0000'1111'1111'0000'0000'0000'0000'1111'1111u;
	//iv0b0000'0000'0001'1111'0000'0000'0000'0000'1111'1111'0000'0000'0000'0000'1111'1111u;

	iv = (iv *
		0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001'0000'0001u) &
		0b1111'0000'0000'1111'0000'0000'1111'0000'0000'1111'0000'0000'1111'0000'0000'1111u;
	//iv0b0001'0000'0000'1111'0000'0000'1111'0000'0000'1111'0000'0000'1111'0000'0000'1111u;

	iv = (iv *
		0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001'0001u) &
		0b0011'0000'1100'0011'0000'1100'0011'0000'1100'0011'0000'1100'0011'0000'1100'0011u;
	//iv0b0001'0000'1100'0011'0000'1100'0011'0000'1100'0011'0000'1100'0011'0000'1100'0011u;

	iv = (iv *
		0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0101u) &
		0b1001'0010'0100'1001'0010'0100'1001'0010'0100'1001'0010'0100'1001'0010'0100'1001u;
	//iv0b0001'0010'0100'1001'0010'0100'1001'0010'0100'1001'0010'0100'1001'0010'0100'1001u;

	return iv;
}
#else
__device__ HashType ExpandBits(HashType iv) {
	iv = (iv * 0x00010001u) &
		0xFF0000FFu;
	iv = (iv * 0x00000101u) &
		0x0F00F00Fu;
	iv = (iv * 0x00000011u) &
		0xC30C30C3u;
	iv = (iv * 0x00000005u) &
		0x49249249u;
	return iv;
}
#endif

__global__ void AssignLeafNodes(int iSampleSize, LeafNode* ipLeafNodes,
	InternalNode* ipInternalNodes, int* iSortedObjectIDs, BBox* iBBoxLeaf) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < iSampleSize) {
		new(&ipLeafNodes[idx]) LeafNode();
		if (idx < iSampleSize - 1) {
			new(&ipInternalNodes[idx]) InternalNode();
		}
		ipLeafNodes[idx].SetObjID(iSortedObjectIDs[idx]);
		ipLeafNodes[idx].SetBBox(&iBBoxLeaf[iSortedObjectIDs[idx]]);

#ifdef DEBUG
		if (idx < 50) {
			const BBox* pBBox = ipLeafNodes[idx].GetBBox();
			CUDA_DEBUG_PRINT("iSampleSize=%d, iBBoxLeaf=%p, iSortedObjectIDs[%d]=%d, &iBBoxLeaf[iSortedObjectIDs[%d]]=%p, pBBox=%p\n",
				iSampleSize, iBBoxLeaf, idx, iSortedObjectIDs[idx], idx, &iBBoxLeaf[iSortedObjectIDs[idx]], pBBox);
		}
#endif
	}
}

__global__ void AssignInternalNodes(int iSampleSize, LeafNode* ipLeafNodes,
	InternalNode* ipInternalNodes, HashType* iSortedMortons)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < iSampleSize - 1) {
		int2 range = DetermineRange(iSortedMortons, iSampleSize, idx);
		// Determine where to split the range.
		int split = FindSplit(iSortedMortons, range.x, range.y);

		Node *childL = nullptr, *childR = nullptr;
		NodeType childTypeL, childTypeR;
		// Select childL.
		if (split == range.x) {
			childL = &ipLeafNodes[split];
			childTypeL = LEAFNODE;
		} else {
			childL = &ipInternalNodes[split];
			childTypeL = INTERNALNODE;
		}
		// Select childR.
		if (split + 1 == range.y) {
			childR = &ipLeafNodes[split + 1];
			childTypeR = LEAFNODE;
		} else {
			childR = &ipInternalNodes[split + 1];
			childTypeR = INTERNALNODE;
		}
		// Record parent-child relationships.
		ipInternalNodes[idx].SetChildL(split, childTypeL);
		ipInternalNodes[idx].SetChildR(split + 1, childTypeR);
		childL->SetParentID(idx);
		childR->SetParentID(idx);

#ifdef DEBUG
		if (idx < 50) {
			const BBox* pBBox = ipLeafNodes[idx].GetBBox();
			CUDA_DEBUG_PRINT("iSampleSize=%d, iSortedMortons[%d]=%llu, pBBox=%p\n",
				iSampleSize, idx, iSortedMortons[idx], pBBox);
		}
#endif
	}
}

__device__ int2 DetermineRange(HashType* iSortedMortons, int iNumObjects, int idx) {
	// d should only take the value of 1 or -1
	int dir = Sign(CommonLeadingBits(iSortedMortons, idx, idx + 1, iNumObjects) -
		CommonLeadingBits(iSortedMortons, idx, idx - 1, iNumObjects));
	int dmin = CommonLeadingBits(iSortedMortons, idx, idx - dir, iNumObjects);
	int lmax = 2;
	while (CommonLeadingBits(iSortedMortons, idx, idx + lmax * dir, iNumObjects) > dmin) {
		lmax = lmax * 2;
	}
	int l = 0;
	for (int t = lmax / 2; t >= 1; t /= 2) {
		if (CommonLeadingBits(iSortedMortons, idx, idx + (l + t) * dir, iNumObjects) > dmin) {
			l += t;
		}
	}
	int j = idx + l * dir;
	if (dir > 0) {
		return int2{ idx, j };
	} else {
		return int2{ j, idx };
	}
}

__device__ int Sign(int ix) {
	return (ix > 0) - (ix < 0);
}

// number of common leading bits of two codes
__device__ int CommonLeadingBits(HashType* iSortedMortons, int ix, int iy, int iNumObjects) {
	if (ix >= 0 && ix < iNumObjects && iy >= 0 && iy < iNumObjects) {
#if HASH_64
		return __clzll(iSortedMortons[ix] ^ iSortedMortons[iy]);
#else
		return __clz(iSortedMortons[ix] ^ iSortedMortons[iy]);
#endif
	}
	return -1;
}

// Return the highest position that shares more than commonPrefix bits
__device__ int FindSplit(HashType* iSortedMortons, int iFirst, int iLast) {
	// Identical Morton codes => split the range in the middle.
	HashType firstCode = iSortedMortons[iFirst];
	HashType lastCode = iSortedMortons[iLast];
	if (firstCode == lastCode) {
		return (iFirst + iLast) >> 1;
	}
	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.
#if HASH_64
	int commonPrefix = __clzll(firstCode ^ lastCode);
#else
	int commonPrefix = __clz(firstCode ^ lastCode);
#endif
	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.
	int split = iFirst; // initial guess
	int step = iLast - iFirst;
	do {
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position
		if (newSplit < iLast) {
			HashType splitCode = iSortedMortons[newSplit];
#if HASH_64
			int splitPrefix = __clzll(firstCode ^ splitCode);
#else
			int splitPrefix = __clz(firstCode ^ splitCode);
#endif
			if (splitPrefix > commonPrefix) {
				split = newSplit; // accept proposal
			}
		}
	} while (step > 1);
	return split;
}

__global__ void ValuesKernel(int iSampleSize, int* keys) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < iSampleSize) {
		keys[index] = index;
	}
}

__global__ void InternalNodeBBox(int iSampleSize, LeafNode* ipLeafNodes,
	InternalNode* ipInternalNodes, BBox* iBBoxInner, int* iReadyFlags
)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= iSampleSize) {
		return;
	}
	int parentIdx = ipLeafNodes[idx].ParentID();
	while (parentIdx != -1) {
		InternalNode* pParent = ipInternalNodes + parentIdx;
		const BBox *pBoxL = nullptr, *pBoxR = nullptr;
		if (pParent->ChildTypeL() == INTERNALNODE) {
			pBoxL = ipInternalNodes[pParent->ChildIdxL()].GetBBox();
		} else {
			pBoxL = ipLeafNodes[pParent->ChildIdxL()].GetBBox();
		}
		if (pParent->ChildTypeR() == INTERNALNODE) {
			pBoxR = ipInternalNodes[pParent->ChildIdxR()].GetBBox();
		} else {
			pBoxR = ipLeafNodes[pParent->ChildIdxR()].GetBBox();
		}
		if (pBoxL && pBoxR) {
			if (atomicAdd(&iReadyFlags[parentIdx], 1) == 0) {
				break;
			}
			new(&iBBoxInner[parentIdx]) BBox(*pBoxL);
			iBBoxInner[parentIdx].Union(*pBoxR);
			pParent->SetBBox(&iBBoxInner[parentIdx]);
			parentIdx = pParent->ParentID();
		} else {
			;
		}
	}
#ifdef DEBUG
	const BBox* pBBox = ipInternalNodes[idx].GetBBox();
	if (idx < 50) {
		CUDA_DEBUG_PRINT("iSampleSize=%d, idx=%d, pBBox=%p\n", iSampleSize, idx, pBBox);
	}
#endif
}

__global__ void CompleteBBox(int iSampleSize, BBox* iBBoxLeaf, BBox* iBBoxInner) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= iSampleSize) {
		return;
	}
	iBBoxLeaf[idx].Complete();
	if (idx < iSampleSize - 1) {
		iBBoxInner[idx].Complete();
	}
}

__host__ __device__ bool IntersectRayAABB(const float3& start,
	const float3& dir, const float3& bmin, const float3& bmax, float& t)
{
	//! calculate candidate plane on each axis
	float tx = -1.0f, ty = -1.0f, tz = -1.0f;
	bool inside = true;
	//! use unrolled loops
	//! x
	if (start.x < bmin.x) {
		if (dir.x != 0.0f)
			tx = (bmin.x - start.x) / dir.x;
		inside = false;
	}
	else if (start.x > bmax.x) {
		if (dir.x != 0.0f)
			tx = (bmax.x - start.x) / dir.x;
		inside = false;
	}
	//! y
	if (start.y < bmin.y) {
		if (dir.y != 0.0f)
			ty = (bmin.y - start.y) / dir.y;
		inside = false;
	}
	else if (start.y > bmax.y) {
		if (dir.y != 0.0f)
			ty = (bmax.y - start.y) / dir.y;
		inside = false;
	}
	//! z
	if (start.z < bmin.z) {
		if (dir.z != 0.0f)
			tz = (bmin.z - start.z) / dir.z;
		inside = false;
	}
	else if (start.z > bmax.z) {
		if (dir.z != 0.0f)
			tz = (bmax.z - start.z) / dir.z;
		inside = false;
	}
	//! if point inside all planes
	if (inside) {
		t = 0.0f;
		return true;
	}
	//! we now have t values for each of possible intersection planes
	//! find the maximum to get the intersection point
	float tmax = tx;
	int taxis = 0;
	if (ty > tmax) {
		tmax = ty;
		taxis = 1;
	}
	if (tz > tmax) {
		tmax = tz;
		taxis = 2;
	}
	if (tmax < 0.0f)
		return false;
	//! check that the intersection point lies on the plane we picked
	//! we don't test the axis of closest intersection for precision reasons
	//! no eps for now
	float eps = 0.0f;
	float3 hit = make_float3
	(start.x + dir.x * tmax, start.y + dir.y * tmax, start.z + dir.z * tmax);
	if ((hit.x < bmin.x - eps || hit.x > bmax.x + eps) && taxis != 0)
		return false;
	if ((hit.y < bmin.y - eps || hit.y > bmax.y + eps) && taxis != 1)
		return false;
	if ((hit.z < bmin.z - eps || hit.z > bmax.z + eps) && taxis != 2)
		return false;
	//! output results
	t = tmax;
	return true;
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 normalize(const float3& v) {
	float invLen = 1.0f / sqrtf(dot(v, v));
	return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ __host__ float3 f3_add(float3 A, float3 B) {
	float3 res;
	res.x = A.x + B.x;
	res.y = A.y + B.y;
	res.z = A.z + B.z;
	return res;
}

__device__ __host__ float3 f3_sub(float3 A, float3 B) {
	float3 res;
	res.x = A.x - B.x;
	res.y = A.y - B.y;
	res.z = A.z - B.z;
	return res;
}

__device__ __host__ float f3_dot(float3 A, float3 B) {
	float res;
	res = A.x * B.x + A.y * B.y + A.z * B.z;
	return res;
}

__device__ __host__ float3 f3_crss(float3 A, float3 B) {
	float3 res;
	res.x = A.y * B.z - A.z * B.y;
	res.y = A.z * B.x - A.x * B.z;
	res.z = A.x * B.y - A.y * B.x;
	return res;
}

__device__ __host__ float3 f3_sclrmult(float val, float3 A) {
	float3 res;
	res.x = val * A.x;
	res.y = val * A.y;
	res.z = val * A.z;
	return res;
}



// Moller and Trumbore's method
__host__ __device__ bool IntersectRayTriTwoSided(
	const float3& p, const float3& dir,
	const float3& a, const float3& b, const float3& c,
	float& t, float& u, float& v)
{
	float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
	float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
	//     float3 n = normalize(cross(ab, ac));
	float3 n = (cross(ab, ac));
	float3 ndir = make_float3(-dir.x, -dir.y, -dir.z);
	float d = dot(ndir, n);
	// No need to check for division by zero here as infinity aritmetic will save us...
	float ood = 1.0f / d;
	float3 ap = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);
	t = dot(ap, n) * ood;
	//     cout << "t = " << t << endl;
	if (t < 0.0f)
		return false;
	float3 e = cross(ndir, ap);
	v = dot(ac, e) * ood;
	//     cout << "v = " << v << " | " << dot(ac, e) << " * " << ood << endl;
	if (v < 0.0f || v > 1.0f) {// ...here...
		return false;
	}
	float w = -dot(ab, e) * ood;
	//     cout << "w = " << w << " | " << -dot(ab, e) << " * " << ood << endl;
	if (w < 0.0f || v + w > 1.0f) {// ...and here
		return false;
	}
	u = 1.0f - v - w;
	return true;
}

__global__ void intersectKernel(int iSampleSize2, BBox* iBBoxLeaf2, Triangle* iTriangle2, int* oHit2,
	int* oHit, int iSampleSize, InternalNode* iInternalNodes, LeafNode* iLeafNodes, Triangle* myMesh)
{
	int idx2 = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx2 >= iSampleSize2) {
		return;
	}
	int hit2 = 0;
	const BBox& box2 = iBBoxLeaf2[idx2];
	const Triangle& triangle2 = iTriangle2[idx2];
	if (iSampleSize == 1) {
		int hit = 0;
		const BBox& box = *iLeafNodes[0].GetBBox();
		if (box2.Intersect(box)) {
			//const Triangle& triangle = myMesh[0];
			//if (triangle2.Intersect(triangle)) {
				hit2 = 1;
				hit = 1;
			//}
		}
		oHit2[idx2] = hit2;
		oHit[0] = hit;
		return;
	}

	int visit_stack[64] = { 0 };
	int stack_ptr = 1;
	while (stack_ptr > 0) {
		--stack_ptr;
		int idx = visit_stack[stack_ptr];
		const BBox& box = *iInternalNodes[idx].GetBBox();
		if (!box2.Intersect(box)) {
			continue;
		} else {
		}
		//
		int childIdxL = iInternalNodes[idx].ChildIdxL();
		int childIdxR = iInternalNodes[idx].ChildIdxR();
		//
		if (iInternalNodes[idx].ChildTypeL() == LEAFNODE) {
			const BBox& boxL = *iLeafNodes[childIdxL].GetBBox();
			if (box2.Intersect(boxL)) {
				int objID = iLeafNodes[childIdxL].GetObjID();
				const Triangle& triangle = myMesh[objID];
				if (triangle2.Intersect(triangle)) {
					oHit[objID] = 1;
					hit2 = 1;
				}
			}
		} else {
			visit_stack[stack_ptr] = childIdxL;
			++stack_ptr;
		}
		if (iInternalNodes[idx].ChildTypeR() == LEAFNODE) {
			const BBox& boxR = *iLeafNodes[childIdxR].GetBBox();
			if (box2.Intersect(boxR)) {
				int objID = iLeafNodes[childIdxR].GetObjID();
				const Triangle& triangle = myMesh[objID];
				if (triangle2.Intersect(triangle)) {
					oHit[objID] = 1;
					hit2 = 1;
				}
			}
		} else {
			visit_stack[stack_ptr] = childIdxR;
			++stack_ptr;
		}
	}
	oHit2[idx2] = hit2;


#ifdef DEBUG
	if (idx2 < 50) {
		CUDA_DEBUG_PRINT("idx2=%d, oHit2[%d]=%d, oHit[%d]=%d\n",
			idx2, idx2, oHit2[idx2], idx2, oHit[idx2]);
	}
#endif
}

void CudaBVH::generateValues(int* iKeys) {
	int* d_keys;
	int size = _SampleSize * sizeof(HashType);
	checkCudaErrors(cudaMalloc((void**)&d_keys, size));
	ValuesKernel<<<(_SampleSize+_ThreadsPerBlock-1)/_ThreadsPerBlock, _ThreadsPerBlock>>>(_SampleSize, d_keys);
	getLastCudaError("value err");
	checkCudaErrors(cudaMemcpy(iKeys, d_keys, size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_keys));
}

void CudaBVH::boxIntersect(int iSampleSize2, BBox* iBBoxLeaf2, Triangle* iTriangle2, int* oHit2, int* oHit) {
	cudaMemset((void*)oHit, 0, _SampleSize * sizeof(int));
	cudaMemset((void*)oHit2, 0, iSampleSize2 * sizeof(int));

	// Compute intersection
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	intersectKernel<<<(iSampleSize2 +_ThreadsPerBlock-1)/_ThreadsPerBlock, _ThreadsPerBlock>>>(
		iSampleSize2, iBBoxLeaf2, iTriangle2, oHit2, oHit, _SampleSize, d_internalNodes, d_leafNodes, d_mesh);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	CUDA_DEBUG_PRINT("It took me %f milliseconds to compute intersection.\n", milliseconds);
}



//bool CudaBVH::intersect(const float3& ray_orig, const float3& ray_dir,
//	float& outT, float& outU, float& outV, int& outIdx)
//{
//	float3* d_rayorig;
//	float3* d_raydir;
//	float* d_t;
//	float* d_u;
//	float* d_v;
//	int* d_idx;
//	int* d_hit;
//	checkCudaErrors(cudaMallocManaged((void**)&d_rayorig, sizeof(float3)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_raydir, sizeof(float3)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_t, sizeof(float)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_u, sizeof(float)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_v, sizeof(float)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_idx, sizeof(int)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_hit, sizeof(int)));
//	d_rayorig[0] = ray_orig;
//	d_raydir[0] = ray_dir;
//	checkCudaErrors(cudaDeviceSynchronize());
//	intersectKernel <<<1, 1 >>> (1, d_rayorig, d_raydir, d_t, d_u, d_v,
//		d_idx, d_hit, d_myTree.internalNodes, d_myTree.leafNodes, d_mesh);
//	checkCudaErrors(cudaDeviceSynchronize());
//	outT = d_t[0];
//	outU = d_u[0];
//	outV = d_v[0];
//	outIdx = d_idx[0];
//	int outHit = d_hit[0];
//	checkCudaErrors(cudaFree(d_rayorig));
//	checkCudaErrors(cudaFree(d_raydir));
//	checkCudaErrors(cudaFree(d_t));
//	checkCudaErrors(cudaFree(d_u));
//	checkCudaErrors(cudaFree(d_v));
//	checkCudaErrors(cudaFree(d_idx));
//	checkCudaErrors(cudaFree(d_hit));
//	return outHit;
//}