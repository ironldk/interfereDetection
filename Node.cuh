#ifndef NODE_CUH_
#define NODE_CUH_

#include "BBox.cuh"

//enum NodeType { LEAFNODE, INTERNALNODE };
#define NodeType int
#define LEAFNODE 0
#define INTERNALNODE 1

class Node {
public:
	__host__ __device__ Node(NodeType iType, int iPID)
:
	type(iType),
	_ParentID(iPID),
	_pd_BBox(nullptr)
	{}

	NodeType GetType() {
		return type;
	}

	__host__ __device__ int ParentID() {
		return _ParentID;
	}

	__host__ __device__ void SetParentID(int i) {
		_ParentID = i;
	}

	__host__ __device__ const BBox* GetBBox() {
		return _pd_BBox;
	}

	__host__ __device__ void SetBBox(const BBox* ipBBox) {
		_pd_BBox = ipBBox;
	}

private:
	NodeType type;
	int _ParentID;
	const BBox *_pd_BBox;
};

#endif