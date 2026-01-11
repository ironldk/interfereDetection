#ifndef INTERNALNODE_CUH_
#define INTERNALNODE_CUH_

#include "Node.cuh"

class InternalNode : public Node {
public:
	__host__ __device__ InternalNode()
:
	Node(INTERNALNODE, -1)
{}

	__host__ __device__ void SetChildL(int i, NodeType t) {
		_ChildIdxL = i;
		_ChildTypeL = t;
	}

	__host__ __device__ void SetChildR(int i, NodeType t) {
		_ChildIdxR = i;
		_ChildTypeR = t;
	}

	__host__ __device__ NodeType ChildTypeL() {
		return _ChildTypeL;
	}

	__host__ __device__ NodeType ChildTypeR() {
		return _ChildTypeR;
	}

	__host__ __device__ int ChildIdxL() {
		return _ChildIdxL;
	}

	__host__ __device__ int ChildIdxR() {
		return _ChildIdxR;
	}

private:
	NodeType _ChildTypeL;
	NodeType _ChildTypeR;
	int _ChildIdxL;
	int _ChildIdxR;
};

#endif