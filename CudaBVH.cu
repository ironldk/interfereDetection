#include "CudaBVH.cuh"
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>
using std::cout;
using std::endl;

cub::CachingDeviceAllocator g_allocator(true);

CudaBVH::CudaBVH(std::vector<Triangle>* iTris, BBox* iBBox, int iSampleSize,
	int iThreadsPerBlock, float iMatrixModel[16], float3 *oGL_d_mesh, float3 *oGL_d_box)
:	_SampleSize(iSampleSize),
	_ThreadsPerBlock(iThreadsPerBlock),
	d_mesh(nullptr),
	d_BBoxLeaf(nullptr),
	d_BBoxInner(nullptr),
	d_leafNodes(nullptr),
	d_internalNodes(nullptr)
{
	for (int i = 0; i < 5; ++i) {
		_TimeCost[i][0] = 0.0;
		_TimeCost[i][1] = 0.0;
	}

// generate morton code
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	cub::DoubleBuffer<HashType> d_sortedKeys;
	cub::DoubleBuffer<int> d_sortedValues;
	CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sortedKeys.d_buffers[0], sizeof(HashType) * _SampleSize));
	CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sortedKeys.d_buffers[1], sizeof(HashType) * _SampleSize));
	CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sortedValues.d_buffers[0], sizeof(int) * _SampleSize));
	CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sortedValues.d_buffers[1], sizeof(int) * _SampleSize));

	HashType *d_objKeys = d_sortedKeys.Current();
	int *d_objValues = d_sortedValues.Current();
	d_mesh = reinterpret_cast<Triangle *>(oGL_d_mesh);
	d_BBoxLeaf = reinterpret_cast<BBox *>(oGL_d_box);
	d_BBoxInner = reinterpret_cast<BBox *>(oGL_d_box) + _SampleSize;
	float *d_matrixModel = nullptr;
	float h_min[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, *d_min = nullptr;
	iBBox->ComputeMin(iMatrixModel, h_min);
	CubDebugExit(g_allocator.DeviceAllocate((void **)&d_matrixModel, 16 * sizeof(float)));
	CubDebugExit(g_allocator.DeviceAllocate((void **)&d_min, 3 * sizeof(float)));
	cudaMemcpy(d_mesh, &(*iTris)[0], _SampleSize * sizeof(Triangle), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixModel, iMatrixModel, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_min, h_min, 3 * sizeof(float), cudaMemcpyHostToDevice);

#ifdef DEBUG
	Triangle* h_mesh = new Triangle[_SampleSize];
	cudaMemcpy(h_mesh, d_mesh, _SampleSize * sizeof(Triangle), cudaMemcpyDeviceToHost);
	CUDA_DEBUG_PRINT("iMatrixModel=%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
		iMatrixModel[ 0], iMatrixModel[ 1], iMatrixModel[ 2], iMatrixModel[ 3],
		iMatrixModel[ 4], iMatrixModel[ 5], iMatrixModel[ 6], iMatrixModel[ 7],
		iMatrixModel[ 8], iMatrixModel[ 9], iMatrixModel[10], iMatrixModel[11],
		iMatrixModel[12], iMatrixModel[13], iMatrixModel[14], iMatrixModel[15]
	);
	for (int idx = 0; idx<DEBUG && idx<_SampleSize; ++idx) {
		CUDA_DEBUG_PRINT("idx=%d, h_mesh[idx]'s a=%.9f,%.9f,%.9f, b=%.9f,%.9f,%.9f, c=%.9f,%.9f,%.9f\n",
			idx, h_mesh[idx].a.x, h_mesh[idx].a.y, h_mesh[idx].a.z,
			h_mesh[idx].b.x, h_mesh[idx].b.y, h_mesh[idx].b.z,
			h_mesh[idx].c.x, h_mesh[idx].c.y, h_mesh[idx].c.z);
	}
	delete[] h_mesh;
	h_mesh = nullptr;

	CUDA_DEBUG_PRINT("Morton3DCuda start=======================================================\n");
#endif

	Morton3DCuda<<<(_SampleSize + _ThreadsPerBlock - 1) / _ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_mesh, d_BBoxLeaf, d_matrixModel, d_objKeys, d_objValues, d_min);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	_TimeCost[0][0] += milliseconds;
	_TimeCost[0][1] += 1;

#ifdef DEBUG
	h_mesh = new Triangle[_SampleSize];
	BBox* h_BBoxLeaf = new BBox[_SampleSize];
	HashType* h_objKeys = new HashType[_SampleSize];
	cudaMemcpy(h_mesh, d_mesh, _SampleSize * sizeof(Triangle), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_BBoxLeaf, d_BBoxLeaf, _SampleSize * sizeof(BBox), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_objKeys, d_objKeys, _SampleSize * sizeof(HashType), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_min, d_min, 3 * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_DEBUG_PRINT("iMatrixModel=%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
		iMatrixModel[ 0], iMatrixModel[ 1], iMatrixModel[ 2], iMatrixModel[ 3],
		iMatrixModel[ 4], iMatrixModel[ 5], iMatrixModel[ 6], iMatrixModel[ 7],
		iMatrixModel[ 8], iMatrixModel[ 9], iMatrixModel[10], iMatrixModel[11],
		iMatrixModel[12], iMatrixModel[13], iMatrixModel[14], iMatrixModel[15]
	);
	for (int idx = 0; idx<DEBUG && idx<_SampleSize; ++idx) {
		CUDA_DEBUG_PRINT("idx=%d, h_mesh[idx]'s a=%.9f,%.9f,%.9f, b=%.9f,%.9f,%.9f, c=%.9f,%.9f,%.9f, "
			"h_BBoxLeaf[idx]'s _min=%.9f,%.9f,%.9f, _max=%.9f,%.9f,%.9f, "
#if HASH_64
			"h_objKeys[idx]=%llu\n",
#else
			"h_objKeys[idx]=%u\n",
#endif
			idx, h_mesh[idx].a.x, h_mesh[idx].a.y, h_mesh[idx].a.z,
			     h_mesh[idx].b.x, h_mesh[idx].b.y, h_mesh[idx].b.z,
			     h_mesh[idx].c.x, h_mesh[idx].c.y, h_mesh[idx].c.z,
			h_BBoxLeaf[idx]._min.x, h_BBoxLeaf[idx]._min.y, h_BBoxLeaf[idx]._min.z,
			h_BBoxLeaf[idx]._max.x, h_BBoxLeaf[idx]._max.y, h_BBoxLeaf[idx]._max.z,
			h_objKeys[idx]
		);
	}
	CUDA_DEBUG_PRINT("h_min=%.9f, h_ymin=%.9f, h_zmin=%.9f\n", h_min[0], h_min[1], h_min[2]);

	delete[] h_mesh;
	delete[] h_BBoxLeaf;
	delete[] h_objKeys;
	h_mesh = nullptr;
	h_BBoxLeaf = nullptr;
	h_objKeys = nullptr;

	CUDA_DEBUG_PRINT("Morton3DCuda end=========================================================\n");
#endif

	CubDebugExit(g_allocator.DeviceFree(d_matrixModel));
	CubDebugExit(g_allocator.DeviceFree(d_min));
	d_matrixModel = nullptr;
	d_min = nullptr;


// Radix sort
	// Allocate temporary storage
	size_t temp_storage_bytes = 0;
	void *d_temp_storage = nullptr;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortedKeys, d_sortedValues, _SampleSize));
	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortedKeys, d_sortedValues, _SampleSize));
	cudaEventRecord(stop);

	CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
	d_temp_storage = nullptr;
	d_objKeys = d_sortedKeys.Current();
	d_objValues = d_sortedValues.Current();

	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	_TimeCost[1][0] += milliseconds;
	_TimeCost[1][1] += 1;

#ifdef DEBUG
	h_objKeys = new HashType[_SampleSize];
	int* h_objValues = new int[_SampleSize];
	cudaMemcpy(h_objKeys, d_objKeys, _SampleSize * sizeof(HashType), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_objValues, d_objValues, _SampleSize * sizeof(int), cudaMemcpyDeviceToHost);
	for (int idx = 0; idx<DEBUG && idx<_SampleSize; ++idx) {
#if HASH_64
		CUDA_DEBUG_PRINT("idx=%d, h_objKeys[idx]=%llu, h_objValues[idx]=%d\n",
			idx, h_objKeys[idx], h_objValues[idx]);
#else
		CUDA_DEBUG_PRINT("idx=%d, h_objKeys[idx]=%u, h_objValues[idx]=%d\n",
			idx, h_objKeys[idx], h_objValues[idx]);
#endif
	}
	delete[] h_objKeys;
	delete[] h_objValues;
	h_objKeys = nullptr;
	h_objValues = nullptr;
	CUDA_DEBUG_PRINT("DeviceRadixSort end======================================================\n");
#endif

// Generate Hierachy
	// Construct leaf nodes.
	// Note: This step can be avoided by storing the tree in a slightly different way.
	cudaMalloc((void **)&d_leafNodes, _SampleSize * sizeof(LeafNode));
	cudaMalloc((void **)&d_internalNodes, (_SampleSize-1) * sizeof(InternalNode));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	AssignLeafNodes<<<(_SampleSize + _ThreadsPerBlock - 1) / _ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_leafNodes, d_internalNodes, d_objValues, d_BBoxLeaf);

#ifdef DEBUG
	LeafNode* h_leafNodes = new LeafNode[_SampleSize];
	cudaMemcpy(h_leafNodes, d_leafNodes, _SampleSize * sizeof(LeafNode), cudaMemcpyDeviceToHost);
	for (int idx = 0; idx<DEBUG && idx<_SampleSize; ++idx) {
		CUDA_DEBUG_PRINT("idx=%d, h_leafNodes[idx]._ObjID=%d\n", idx, h_leafNodes[idx].GetObjID());
	}
	delete[] h_leafNodes;
	h_leafNodes = nullptr;

	CUDA_DEBUG_PRINT("AssignLeafNodes end======================================================\n");
#endif

	AssignInternalNodes<<<(_SampleSize + _ThreadsPerBlock - 1) / _ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_leafNodes, d_internalNodes, d_objKeys);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	_TimeCost[2][0] += milliseconds;
	_TimeCost[2][1] += 1;

	CubDebugExit(g_allocator.DeviceFree(d_sortedKeys.d_buffers[0]));
	CubDebugExit(g_allocator.DeviceFree(d_sortedKeys.d_buffers[1]));
	CubDebugExit(g_allocator.DeviceFree(d_sortedValues.d_buffers[0]));
	CubDebugExit(g_allocator.DeviceFree(d_sortedValues.d_buffers[1]));

#ifdef DEBUG
	InternalNode* h_internalNodes = new InternalNode[_SampleSize];
	cudaMemcpy(h_internalNodes, d_internalNodes, (_SampleSize-1) * sizeof(InternalNode), cudaMemcpyDeviceToHost);
	for (int idx = 0; idx<DEBUG-1 && idx<_SampleSize-1; ++idx) {
		CUDA_DEBUG_PRINT("idx=%d, h_internalNodes[idx]'s _ParentID=%d, "
			"_ChildTypeL=%d, _ChildIdxL=%d, _ChildTypeR=%d, _ChildIdxR=%d\n",
			idx, h_internalNodes[idx].ParentID(),
			h_internalNodes[idx].ChildTypeL(), h_internalNodes[idx].ChildIdxL(),
			h_internalNodes[idx].ChildTypeR(), h_internalNodes[idx].ChildIdxR()
		);
	}

	CUDA_DEBUG_PRINT("AssignInternalNodes end==================================================\n");
#endif

// Assign bounding box to internal nodes
	int *d_ReadyFlags = nullptr;
	CubDebugExit(g_allocator.DeviceAllocate((void **)&d_ReadyFlags, (_SampleSize - 1) * sizeof(int)));
	cudaMemset(d_ReadyFlags, 0, (_SampleSize - 1) * sizeof(int));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	InternalNodeBBox<<<(_SampleSize + _ThreadsPerBlock - 1) / _ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_leafNodes, d_internalNodes, d_BBoxInner, d_ReadyFlags);

#ifdef DEBUG
	BBox* h_BBoxInner = new BBox[_SampleSize-1];
	cudaMemcpy(h_BBoxInner, d_BBoxInner, (_SampleSize-1) * sizeof(BBox), cudaMemcpyDeviceToHost);
	for (int idx = 0; idx<DEBUG-1 && idx<_SampleSize-1; ++idx) {
		CUDA_DEBUG_PRINT("idx=%d, h_BBoxInner[idx]'s _min=%.9f,%.9f,%.9f, _max=%.9f,%.9f,%.9f\n",
			idx, 
			h_BBoxInner[idx]._min.x, h_BBoxInner[idx]._min.y, h_BBoxInner[idx]._min.z,
			h_BBoxInner[idx]._max.x, h_BBoxInner[idx]._max.y, h_BBoxInner[idx]._max.z);
	}
	delete[] h_BBoxInner;
	h_BBoxInner = nullptr;

	CUDA_DEBUG_PRINT("InternalNodeBBox end=====================================================\n");
#endif

	CompleteBBox<<<(_SampleSize + _ThreadsPerBlock - 1) / _ThreadsPerBlock, _ThreadsPerBlock>>>
		(_SampleSize, d_BBoxLeaf, d_BBoxInner);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	_TimeCost[3][0] += milliseconds;
	_TimeCost[3][1] += 1;

	CubDebugExit(g_allocator.DeviceFree(d_ReadyFlags));
	d_ReadyFlags = nullptr;

#ifdef DEBUG
	cudaDeviceSynchronize();
	CUDA_DEBUG_PRINT("CompleteBBox end=========================================================\n");
	fflush(stdout);
#endif
}

CudaBVH::~CudaBVH() {
	d_mesh = nullptr;
	d_BBoxLeaf = nullptr;
	d_BBoxInner = nullptr;
	cudaFree(d_leafNodes);
	cudaFree(d_internalNodes);
	d_leafNodes = nullptr;
	d_internalNodes = nullptr;
}

__global__ void Morton3DCuda(int iSampleSize, Triangle* iTriangle, BBox* oBBoxLeaf,
	float iMatrixModel[16], HashType *oMCode, int *oObjectIDs,
	float *imin)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= iSampleSize) {
		return;
	}
	float x = 0.0f, y = 0.0f, z = 0.0f;
	iTriangle[idx].Transform(iMatrixModel);
	new (&oBBoxLeaf[idx]) BBox(iTriangle[idx]);
	oBBoxLeaf[idx].GetCenter(x, y, z);
	x -= imin[0];
	y -= imin[1];
	z -= imin[2];

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
	oMCode[idx] = ((ex << 2) + (ey << 1) + ez);
	oObjectIDs[idx] = idx;
}

#if HASH_64
// Expand a 21-bit integer into 63 bits by inserting 2 zeros before each bit.
__device__ HashType ExpandBits(HashType iv) {
	// iv0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001'1111'1111'1111'1111'1111u;

	iv = (iv *
		 0b0000'0000'0000'0000'0000'0000'0000'0001'0000'0000'0000'0000'0000'0000'0000'0001u) &
		 0b1111'1111'1111'1111'0000'0000'0000'0000'0000'0000'0000'0000'1111'1111'1111'1111u;
	// iv0b0000'0000'0001'1111'0000'0000'0000'0000'0000'0000'0000'0000'1111'1111'1111'1111u;

	iv = (iv *
		 0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001'0000'0000'0000'0001u) &
		 0b0000'0000'1111'1111'0000'0000'0000'0000'1111'1111'0000'0000'0000'0000'1111'1111u;
	// iv0b0000'0000'0001'1111'0000'0000'0000'0000'1111'1111'0000'0000'0000'0000'1111'1111u;

	iv = (iv *
		 0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001'0000'0001u) &
		 0b1111'0000'0000'1111'0000'0000'1111'0000'0000'1111'0000'0000'1111'0000'0000'1111u;
	// iv0b0001'0000'0000'1111'0000'0000'1111'0000'0000'1111'0000'0000'1111'0000'0000'1111u;

	iv = (iv *
		 0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001'0001u) &
		 0b0011'0000'1100'0011'0000'1100'0011'0000'1100'0011'0000'1100'0011'0000'1100'0011u;
	// iv0b0001'0000'1100'0011'0000'1100'0011'0000'1100'0011'0000'1100'0011'0000'1100'0011u;

	iv = (iv *
		 0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0101u) &
		 0b1001'0010'0100'1001'0010'0100'1001'0010'0100'1001'0010'0100'1001'0010'0100'1001u;
	// iv0b0001'0010'0100'1001'0010'0100'1001'0010'0100'1001'0010'0100'1001'0010'0100'1001u;

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

__global__ void AssignLeafNodes(int iSampleSize, LeafNode *ipLeafNodes,
	InternalNode *ipInternalNodes, int *iSortedObjectIDs, BBox *iBBoxLeaf)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= iSampleSize) {
		return;
	}
	new (&ipLeafNodes[idx]) LeafNode();
	if (idx < iSampleSize - 1) {
		new (&ipInternalNodes[idx]) InternalNode();
	}
	ipLeafNodes[idx].SetObjID(iSortedObjectIDs[idx]);
	ipLeafNodes[idx].SetBBox(&iBBoxLeaf[iSortedObjectIDs[idx]]);
}

__global__ void AssignInternalNodes(int iSampleSize, LeafNode *ipLeafNodes,
	InternalNode *ipInternalNodes, HashType *iSortedMortons)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= iSampleSize - 1) {
		return;
	}
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
}

__device__ int2 DetermineRange(HashType *iSortedMortons, int iNumObjects, int idx) {
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
		return int2{idx, j};
	} else {
		return int2{j, idx};
	}
}

__device__ int Sign(int ix) {
	return (ix > 0) - (ix < 0);
}

// number of common leading bits of two codes
__device__ int CommonLeadingBits(HashType *iSortedMortons, int ix, int iy, int iNumObjects) {
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
__device__ int FindSplit(HashType *iSortedMortons, int iFirst, int iLast) {
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
		step = (step + 1) >> 1;		 // exponential decrease
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

__global__ void InternalNodeBBox(int iSampleSize, LeafNode *ipLeafNodes,
	InternalNode *ipInternalNodes, BBox *iBBoxInner, int *iReadyFlags)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= iSampleSize) {
		return;
	}
	int parentIdx = ipLeafNodes[idx].ParentID();
	while (parentIdx != -1) {
		InternalNode *pParent = ipInternalNodes + parentIdx;
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
			new (&iBBoxInner[parentIdx]) BBox(*pBoxL);
			iBBoxInner[parentIdx].Union(*pBoxR);
			pParent->SetBBox(&iBBoxInner[parentIdx]);
			parentIdx = pParent->ParentID();
		} else {
			;
		}
	}
}

__global__ void CompleteBBox(int iSampleSize, BBox *iBBoxLeaf, BBox *iBBoxInner) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= iSampleSize) {
		return;
	}
	iBBoxLeaf[idx].Complete();
	if (idx < iSampleSize - 1) {
		iBBoxInner[idx].Complete();
	}
}

__global__ void intersectKernel(int iSampleSize2, BBox *iBBoxLeaf2, Triangle *iTriangle2, int *oHit2,
	int *oHit, int iSampleSize, InternalNode *iInternalNodes, LeafNode *iLeafNodes, Triangle *myMesh)
{
	int idx2 = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx2 >= iSampleSize2) {
		return;
	}
	int hit2 = 0;
	const BBox &box2 = iBBoxLeaf2[idx2];
	const Triangle &triangle2 = iTriangle2[idx2];
	if (iSampleSize == 1) {
		int hit = 0;
		const BBox &box = *iLeafNodes[0].GetBBox();
		if (box2.Intersect(box)) {
			const Triangle &triangle = myMesh[0];
			if (triangle2.Intersect(triangle)) {
				hit2 = 1;
				hit = 1;
			}
		}
		oHit2[idx2] = hit2;
		oHit[0] = hit;
		return;
	}

	int visit_stack[64] = {0};
	for (int stack_ptr = 0; stack_ptr >= 0;) {
		int idx = visit_stack[stack_ptr--];
		const BBox &box = *iInternalNodes[idx].GetBBox();
		if (!box2.Intersect(box)) {
			continue;
		}
		//
		int childIdxL = iInternalNodes[idx].ChildIdxL();
		int childIdxR = iInternalNodes[idx].ChildIdxR();
		//
		if (iInternalNodes[idx].ChildTypeL() == LEAFNODE) {
			const BBox &boxL = *iLeafNodes[childIdxL].GetBBox();
			if (box2.Intersect(boxL)) {
				int objID = iLeafNodes[childIdxL].GetObjID();
				const Triangle &triangle = myMesh[objID];
				if (triangle2.Intersect(triangle)) {
					oHit[objID] = 1;
					hit2 = 1;
				}
			}
		} else {
			visit_stack[++stack_ptr] = childIdxL;
		}
		if (iInternalNodes[idx].ChildTypeR() == LEAFNODE) {
			const BBox &boxR = *iLeafNodes[childIdxR].GetBBox();
			if (box2.Intersect(boxR)) {
				int objID = iLeafNodes[childIdxR].GetObjID();
				const Triangle &triangle = myMesh[objID];
				if (triangle2.Intersect(triangle)) {
					oHit[objID] = 1;
					hit2 = 1;
				}
			}
		} else {
			visit_stack[++stack_ptr] = childIdxR;
		}
	}
	oHit2[idx2] = hit2;
}

void CudaBVH::boxIntersect(int iSampleSize2, BBox *iBBoxLeaf2, Triangle *iTriangle2, int *oHit2, int *oHit) {
	cudaMemset((void *)oHit, 0, _SampleSize * sizeof(int));
	cudaMemset((void *)oHit2, 0, iSampleSize2 * sizeof(int));

	// Compute intersection
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	intersectKernel<<<(iSampleSize2 + _ThreadsPerBlock - 1) / _ThreadsPerBlock, _ThreadsPerBlock>>>(
		iSampleSize2, iBBoxLeaf2, iTriangle2, oHit2, oHit, _SampleSize, d_internalNodes, d_leafNodes, d_mesh);

#ifdef DEBUG
	cudaError_t err = cudaGetLastError();  // 检查 launch 参数
	if (err != cudaSuccess) {
		printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();
	err = cudaGetLastError();  // 检查执行期错误
	if (err != cudaSuccess) {
		printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
	}
#endif

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	_TimeCost[4][0] += milliseconds;
	_TimeCost[4][1] += 1;

#ifdef DEBUG
	BBox *h_BBoxLeaf2 = new BBox[iSampleSize2];
	int* hHit2 = new int[iSampleSize2];
	cudaMemcpy(h_BBoxLeaf2, iBBoxLeaf2, iSampleSize2 * sizeof(BBox), cudaMemcpyDeviceToHost);
	cudaMemcpy(hHit2, oHit2, iSampleSize2 * sizeof(int), cudaMemcpyDeviceToHost);
	for (int idx2 = 0; idx2<DEBUG && idx2<iSampleSize2; ++idx2) {
		CUDA_DEBUG_PRINT("idx2=%d, h_BBoxLeaf2[idx2]'s _min=%.9f,%.9f,%.9f, _max=%.9f,%.9f,%.9f, oHit2[idx2]=%d\n",
			idx2,
			h_BBoxLeaf2[idx2]._min.x, h_BBoxLeaf2[idx2]._min.y, h_BBoxLeaf2[idx2]._min.z,
			h_BBoxLeaf2[idx2]._max.x, h_BBoxLeaf2[idx2]._max.y, h_BBoxLeaf2[idx2]._max.z,
			hHit2[idx2]
		);
	}
	delete[] h_BBoxLeaf2;
	delete[] hHit2;
	h_BBoxLeaf2 = nullptr;
	hHit2 = nullptr;

	CUDA_DEBUG_PRINT("boxIntersect end============================================================\n");
	fflush(stdout);
	__debugbreak();
#endif
}