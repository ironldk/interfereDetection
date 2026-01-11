#ifndef LEAFNODE_CUH_
#define LEAFNODE_CUH_

#include "Node.cuh"

class LeafNode : public Node {
public:
	__host__ __device__ LeafNode()
:
	Node(LEAFNODE, -1),
	_ObjID(0)
	{}

	__host__ __device__ void SetObjID(int id) {
		_ObjID = id;
	}

	__host__ __device__ int GetObjID() {
		return _ObjID;
	}

private:
	int _ObjID;
};

#endif