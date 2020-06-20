#ifndef INTERNALNODE_CUH_
#define INTERNALNODE_CUH_

#include "Node.cuh"

class InternalNode : public Node {
public:
	__host__ __device__ InternalNode();
	//__host__ __device__ void setType();
	__host__ __device__ void setLeftNode(int i, NodeType t);
	__host__ __device__ void setRightNode(int i, NodeType t);
	__host__ __device__ NodeType leftNodeType();
	__host__ __device__ NodeType rightNodeType();
	__host__ __device__ int leftNodeIdx();
	__host__ __device__ int rightNodeIdx();
private:
	NodeType leftType;
	int leftNodeIdx;
	NodeType rightType;
	int rightNodeIdx;
};

#endif