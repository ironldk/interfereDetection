#include "Node.cuh"

Node::Node(NodeType t, int i, NodeType pType, int p, BBox b)
: type(t), idx(i), parentType(pType), parent(p), box(b) {}

string Node::getType() {
	switch (type) {
	case LEAFNODE:
		return "LEAFNODE";
	case INTERNALNODE:
		return "INTERNALNODE";
	}
}

__host__ __device__ int Node::getIdx() {
	return idx;
}

__host__ __device__ void Node::setIdx(int i) {
	idx = i;
}

string Node::parentType() {
	switch (parentType) {
	case LEAFNODE:
		return "LEAFNODE";
	case INTERNALNODE:
		return "INTERNALNODE";
	}
}

__host__ __device__ int Node::getParent() {
	return parent;
}

__host__ __device__ void Node::setParent(int i, NodeType type) {
	parent = i;
	parentType = type;
}

__host__ __device__ BBox Node::getBBox() {
	return box;
}

__host__ __device__ void Node::setBBox(const BBox& b) {
	box = b;
}