#ifndef LEAFNODE_CUH_
#define LEAFNODE_CUH_

#include "Node.cuh"

class LeafNode : public Node {
public:
	__host__ __device__ LeafNode();
	__host__ __device__ void setObjectID(int id);
	__host__ __device__ int getObjectID();
private:
	int objID;
};

#endif // !LEAFNODE_CUH_