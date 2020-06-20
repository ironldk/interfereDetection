#include "LeafNode.cuh"

__host__ __device__ LeafNode::LeafNode()
: Node(LEAFNODE, 0, INTERNALNODE, 0, BBox()), objID(0) {}

__host__ __device__ void LeafNode::setObjectID(int id) {
    objID = id;
}

__host__ __device__ int LeafNode::getObjectID() {
    return objID;
}