#ifndef NODE_CUH_
#define NODE_CUH_

#include<string>
#include "BBox.cuh"
using std::string;
enum NodeType { LEAFNODE, INTERNALNODE };
class Node {
public:
	Node(NodeType t, int i, NodeType pType, int p, BBox b);
	string getType();
	__host__ __device__ int getIdx();
	__host__ __device__ void setIdx(int i);
	string parentType();
	__host__ __device__ int getParent();
	__host__ __device__ void setParent(int i, NodeType type);
	__host__ __device__ BBox getBBox();
	__host__ __device__ void setBBox(const BBox& b);
private:
	NodeType type;
	int idx;
	NodeType parentType;
	int parent;
	BBox box;
};

#endif // !NODE_CUH_