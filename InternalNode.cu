#include "InternalNode.cuh"

__host__ __device__ InternalNode::InternalNode()
: Node(INTERNALNODE, 0, INTERNALNODE, -1, BBox()) {}
//__host__ __device__ void InternalNode::setType() {
//	type = INTERNALNODE;
//}

__host__ __device__ void InternalNode::setLeftNode(int i, NodeType t) {
    leftNodeIdx = i;
    leftType = t;
}

__host__ __device__ void InternalNode::setRightNode(int i, NodeType t) {
    rightNodeIdx = i;
    rightType = t;
}

__host__ __device__ NodeType InternalNode::leftNodeType() {
    return leftType;
}

__host__ __device__ NodeType InternalNode::rightNodeType() {
    return rightType;
}

__host__ __device__ int InternalNode::leftNodeIdx() {
    return leftNodeIdx;
}

__host__ __device__ int InternalNode::rightNodeIdx() {
    return rightNodeIdx;
}