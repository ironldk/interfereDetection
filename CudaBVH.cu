#include "CudaBVH.cuh"

__device__ int findSplit(HashType* sortedMortonCodes, int first, int last) {
	// Identical Morton codes => split the range in the middle.
	HashType firstCode = sortedMortonCodes[first];
	HashType lastCode = sortedMortonCodes[last];
	if (firstCode == lastCode)
		return (first + last) >> 1;
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
	int split = first; // initial guess
	int step = last - first;
	do {
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position
		if (newSplit < last) {
			HashType splitCode = sortedMortonCodes[newSplit];
#if HASH_64
			int splitPrefix = __clzll(firstCode ^ splitCode);
#else
			int splitPrefix = __clz(firstCode ^ splitCode);
#endif
			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);
	return split;
}

// number of common leading bits of two codes
__device__ int delta(HashType* sortedMortonCodes, int x, int y, int numObjects){
	if (x>=0 && x<numObjects && y>=0 && y<numObjects) {
#if HASH_64
		return __clzll(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
#else
		return __clz(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
#endif
	}
	return -1;
}

__device__ int sign(int x) { return (x > 0) - (x < 0); }

__device__ int2 determineRange(HashType* sortedMortonCodes,
	int numObjects, int idx)
{
	int d = sign(delta(sortedMortonCodes, idx, idx + 1, numObjects) -
		delta(sortedMortonCodes, idx, idx - 1, numObjects));
	int dmin = delta(sortedMortonCodes, idx, idx - d, numObjects);
	int lmax = 2;
	while (delta(sortedMortonCodes, idx, idx + lmax * d, numObjects) > dmin)
		lmax = lmax * 2;
	int l = 0; 
	for (int t = lmax / 2; t >= 1; t /= 2) {
		if (delta(sortedMortonCodes, idx, idx + (l + t) * d, numObjects) > dmin)
			l += t;
	}
	int j = idx + l * d;
	int2 range;
	range.x = min(idx, j);
	range.y = max(idx, j);
	if (idx == 38043 || idx == 38044 || idx == 38045 || idx == 38046 ||
		idx == 38047 || idx == 38048)
		printf("idx %d range :%d - %d j: %d morton: %d\n",
			idx, range.x, range.y, j, sortedMortonCodes[idx]);
	return range;
}

#if HASH_64
__device__ HashType expandBits(HashType v) {
	v = (v * 0x000100000001u) & 0xFFFF00000000FFFFu;
	v = (v * 0x000000010001u) & 0x00FF0000FF0000FFu;
	v = (v * 0x000000000101u) & 0xF00F00F00F00F00Fu;
	v = (v * 0x000000000011u) & 0x30C30C30C30C30C3u;
	v = (v * 0x000000000005u) & 0x9249249249249249u;
	return v;
}
#else
__device__ HashType expandBits(HashType v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}
#endif


__global__ void assignInternalNodes(int SAMPLE_SIZE, HashType* sortedMortonCodes,
	LeafNode* leafNodes, InternalNode* internalNodes, int* sortedObjectIDs)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE - 1) {
		int2 range = determineRange(sortedMortonCodes, SAMPLE_SIZE, idx);
		int first = range.x;
		int last = range.y;
		// Determine where to split the range.
		int split = findSplit(sortedMortonCodes, first, last);
		// Select childA.
		Node* childA;
		int childAIdx;
		NodeType childAType;
		if (split == first) {
			childA = &leafNodes[split];
			childAIdx = split;
			childAType = LEAFNODE;
		}
		else {
			childA = &internalNodes[split];
			childAIdx = split;
			childAType = INTERNALNODE;
		}
		// Select childB.
		Node* childB;
		int childBIdx;
		NodeType childBType;
		if (split + 1 == last) {
			childB = &leafNodes[split + 1];
			childBIdx = split + 1;
			childBType = LEAFNODE;
		}
		else {
			childB = &internalNodes[split + 1];
			childBIdx = split + 1;
			childBType = INTERNALNODE;
		}
		// Record parent-child relationships.
		//internalNodes[idx].setType();
		internalNodes[idx].setLeftNode(childAIdx, childAType);
		internalNodes[idx].setRightNode(childBIdx, childBType);
		internalNodes[idx].setIdx(idx);
		// the initialization is moved outside
		// and using constructors to avoid overwrite
//         if (0)        {
//             //         if ()
//             //             internalNodes[idx].setParent(-1, NODE);
//         }
		childA->setParent(idx, INTERNALNODE);
		childB->setParent(idx, INTERNALNODE);
		//printf("%d %d %d %d %d %d\n",
		//idx, first, last, split, childA->getParent(), childB->getParent());
	}
}

#if 0
__global__ void morton3DCuda(int SAMPLE_SIZE, HashType* c, const BBox* objects) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE) {
		float x, y, z;
		x = (objects[idx]._max.x + objects[idx]._min.x) / 2;
		y = (objects[idx]._max.y + objects[idx]._min.y) / 2;
		z = (objects[idx]._max.z + objects[idx]._min.z) / 2;
		x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
		y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
		z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
		HashType xx = expandBits((HashType)x);
		HashType yy = expandBits((HashType)y);
		HashType zz = expandBits((HashType)z);
		c[idx] = xx * 4 + yy * 2 + zz;
	}
}
#else
__device__ MortonRec::MortonRec(int sample_size, const BBox& bbox, int idx) {
	bool needprint = idx >= 38040 && idx <= 38050;
	bbox.GetCenter(x, y, z);
	//if (needprint) printf("idx: %d  x: %f\n", idx, x);
	//if (needprint) printf("idx: %d  y: %f\n", idx, y);
	//if (needprint) printf("idx: %d  z: %f\n", idx, z);
#if HASH_64
	xx = x * 1024.0f * 1024.0f;
	//if (needprint) printf("idx: %d  xx: %f\n", idx, x);
	yy = y * 1024.0f * 1024.0f;
	//if (needprint) printf("idx: %d  yy: %f\n", idx, y);
	zz = z * 1024.0f * 1024.0f;
	//if (needprint) printf("idx: %d  zz: %f\n", idx, z);
#else
	x = x * 1023.0f;  //if (needprint) printf("idx: %d  xx: %f\n", idx, x);
	y = y * 1023.0f;  //if (needprint) printf("idx: %d  yy: %f\n", idx, y);
	z = z * 1023.0f;  //if (needprint) printf("idx: %d  zz: %f\n", idx, z);
#endif
	ex = expandBits((HashType)((double)xx));
	//if (needprint) printf("idx: %d  expand x: %d\n", idx, xx);
	ey = expandBits((HashType)((double)yy));
	//if (needprint) printf("idx: %d  expand y: %d\n", idx, yy);
	ez = expandBits((HashType)((double)zz));
	//if (needprint) printf("idx: %d  expand z: %d\n", idx, zz);
	m = (ex * 4 + ey * 2 + ez) * sample_size + idx;		// ·ÀÖ¹MortonÂëÖØ¸´
	//if (needprint) printf("idx: %d  hash: %d\n", idx, c[idx]);
}

__global__ void morton3DCuda(int SAMPLE_SIZE, HashType* c, const BBox* objects,
	MortonRec* mor)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE) {
		mor[idx] = MortonRec(SAMPLE_SIZE, objects[idx], idx);
		c[idx] = mor[idx].m;
	}
}
#endif

__global__ void valuesKernel(int SAMPLE_SIZE, int* keys) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < SAMPLE_SIZE)
		keys[index] = index;
}

__global__ void internalNodeBBox(int SAMPLE_SIZE, int* atom,
	InternalNode* internalNodes, LeafNode* leafNodes, BBox* d_myBBox)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE) {
		Node* ptr = &leafNodes[idx];
		InternalNode* parent = &internalNodes[ptr->getParent()];
		while (parent->getIdx() < SAMPLE_SIZE - 1 && parent->getIdx() > -1 &&
			atomicCAS(&atom[parent->getIdx()], 0, 1) == 1)
		{
			BBox leftBox, rightBox;
			//printf("In while %d\n", parent->getIdx());
			if (parent->leftNodeType() == INTERNALNODE) {
				leftBox = internalNodes[parent->leftNodeIdx()].getBBox();
			} else {
				leftBox = leafNodes[parent->leftNodeIdx()].getBBox();
			}
			BBox buf(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX);
			buf.MakeEnvelope(leftBox);
			buf.MakeEnvelope(rightBox);

			parent->setBBox(buf);
			ptr = parent;
			if (ptr->getParent() > -1/* && ptr->getParent() < SAMPLE_SIZE - 1*/)
				parent = &internalNodes[ptr->getParent()];
			else return;
		}
	}
}

__global__ void assignLeafNodes
	(int SAMPLE_SIZE, LeafNode* leafNodes, int* sortedObjectIDs, BBox* bbox)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE) {
		//leafNodes[idx].setType();
		leafNodes[idx].setIdx(idx);
		leafNodes[idx].setObjectID(sortedObjectIDs[idx]);
		leafNodes[idx].setBBox(bbox[sortedObjectIDs[idx]]);
		//printf("Index: %d ObjectID: %d\n", idx, leafNodes[idx].getObjectID());
	}
}

__host__ __device__ bool IntersectRayAABB(const float3& start,
	const float3& dir, const float3& bmin, const float3& bmax, float& t)
{
	//! calculate candidate plane on each axis
	float tx = -1.0f, ty = -1.0f, tz = -1.0f;
	bool inside = true;
	//! use unrolled loops
	//! x
	if (start.x < bmin.x) {
		if (dir.x != 0.0f)
			tx = (bmin.x - start.x) / dir.x;
		inside = false;
	}
	else if (start.x > bmax.x) {
		if (dir.x != 0.0f)
			tx = (bmax.x - start.x) / dir.x;
		inside = false;
	}
	//! y
	if (start.y < bmin.y) {
		if (dir.y != 0.0f)
			ty = (bmin.y - start.y) / dir.y;
		inside = false;
	}
	else if (start.y > bmax.y) {
		if (dir.y != 0.0f)
			ty = (bmax.y - start.y) / dir.y;
		inside = false;
	}
	//! z
	if (start.z < bmin.z) {
		if (dir.z != 0.0f)
			tz = (bmin.z - start.z) / dir.z;
		inside = false;
	}
	else if (start.z > bmax.z) {
		if (dir.z != 0.0f)
			tz = (bmax.z - start.z) / dir.z;
		inside = false;
	}
	//! if point inside all planes
	if (inside) {
		t = 0.0f;
		return true;
	}
	//! we now have t values for each of possible intersection planes
	//! find the maximum to get the intersection point
	float tmax = tx;
	int taxis = 0;
	if (ty > tmax) {
		tmax = ty;
		taxis = 1;
	}
	if (tz > tmax) {
		tmax = tz;
		taxis = 2;
	}
	if (tmax < 0.0f)
		return false;
	//! check that the intersection point lies on the plane we picked
	//! we don't test the axis of closest intersection for precision reasons
	//! no eps for now
	float eps = 0.0f;
	float3 hit = make_float3
	(start.x + dir.x * tmax, start.y + dir.y * tmax, start.z + dir.z * tmax);
	if ((hit.x < bmin.x - eps || hit.x > bmax.x + eps) && taxis != 0)
		return false;
	if ((hit.y < bmin.y - eps || hit.y > bmax.y + eps) && taxis != 1)
		return false;
	if ((hit.z < bmin.z - eps || hit.z > bmax.z + eps) && taxis != 2)
		return false;
	//! output results
	t = tmax;
	return true;
}

__host__ __device__ bool IntersectAABBAABB(const BBox& b, const BBox& a) {
	return a.Intersect(b);
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 normalize(const float3& v) {
	float invLen = 1.0f / sqrtf(dot(v, v));
	return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ __host__ float3 f3_add(float3 A, float3 B) {
	float3 res;
	res.x = A.x + B.x;
	res.y = A.y + B.y;
	res.z = A.z + B.z;
	return res;
}

__device__ __host__ float3 f3_sub(float3 A, float3 B) {
	float3 res;
	res.x = A.x - B.x;
	res.y = A.y - B.y;
	res.z = A.z - B.z;
	return res;
}

__device__ __host__ float f3_dot(float3 A, float3 B) {
	float res;
	res = A.x * B.x + A.y * B.y + A.z * B.z;
	return res;
}

__device__ __host__ float3 f3_crss(float3 A, float3 B) {
	float3 res;
	res.x = A.y * B.z - A.z * B.y;
	res.y = A.z * B.x - A.x * B.z;
	res.z = A.x * B.y - A.y * B.x;
	return res;
}

__device__ __host__ float3 f3_sclrmult(float val, float3 A) {
	float3 res;
	res.x = val * A.x;
	res.y = val * A.y;
	res.z = val * A.z;
	return res;
}

__device__ __host__ float f_clamp(float n, float min, float max) {
	if (n < min) return min;
	if (n > max) return max;
	return n;
}

// Moller and Trumbore's method
__host__ __device__ bool IntersectRayTriTwoSided(
	const float3& p, const float3& dir,
	const float3& a, const float3& b, const float3& c,
	float& t, float& u, float& v)
{
	float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
	float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
	//     float3 n = normalize(cross(ab, ac));
	float3 n = (cross(ab, ac));
	float3 ndir = make_float3(-dir.x, -dir.y, -dir.z);
	float d = dot(ndir, n);
	// No need to check for division by zero here as infinity aritmetic will save us...
	float ood = 1.0f / d;
	float3 ap = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);
	t = dot(ap, n) * ood;
	//     cout << "t = " << t << endl;
	if (t < 0.0f)
		return false;
	float3 e = cross(ndir, ap);
	v = dot(ac, e) * ood;
	//     cout << "v = " << v << " | " << dot(ac, e) << " * " << ood << endl;
	if (v < 0.0f || v > 1.0f) {// ...here...
		return false;
	}
	float w = -dot(ab, e) * ood;
	//     cout << "w = " << w << " | " << -dot(ab, e) << " * " << ood << endl;
	if (w < 0.0f || v + w > 1.0f) {// ...and here
		return false;
	}
	u = 1.0f - v - w;
	return true;
}

__device__ __host__ float edge_to_edge(float3 p1, float3 q1, float3 p2, float3 q2,
	float& s, float& t, float3& c1, float3& c2) {
	float3 d1 = f3_sub(q1, p1); // Direction vector of segment S1
	float3 d2 = f3_sub(q2, p2); // Direction vector of segment S2
	float3 r = f3_sub(p1, p2);
	float a = f3_dot(d1, d1); // Squared length of segment S1, always nonnegative
	float e = f3_dot(d2, d2); // Squared length of segment S2, always nonnegative
	float f = f3_dot(d2, r);
	// Check if either or both segments degenerate into points
	if (a <= EPSILON && e <= EPSILON) {
		// Both segments degenerate into points
		s = t = 0.0f;
		c1 = p1;
		c2 = p2;
		float3 c2c1 = f3_sub(c1, c2);
		return f3_dot(c2c1, c2c1);
	}
	if (a <= EPSILON) {
		// First segment degenerates into a point
		s = 0.0f;
		t = f / e; // s = 0 => t = (b*s + f) / e = f / e
		t = f_clamp(t, 0.0f, 1.0f);
	}
	else {
		float c = f3_dot(d1, r);
		if (e <= EPSILON) {
			// Second segment degenerates into a point
			t = 0.0f;
			s = f_clamp(-c / a, 0.0f, 1.0f); // t = 0 => s = (b*t - c) / a = -c / a
		}
		else {
			// The general nondegenerate case starts here
			float b = f3_dot(d1, d2);
			float denom = a * e - b * b; // Always nonnegative
			// If segments not parallel, compute closest point on L1 to L2 and
			// clamp to segment S1. Else pick arbitrary s (here 0)
			if (denom != 0.0f) {
				s = f_clamp((b * f - c * e) / denom, 0.0f, 1.0f);
			}
			else
				s = 0.0f;
			// Compute point on L2 closest to S1(s) using
			// t = Dot((P1 + D1*s) - P2,D2) / Dot(D2,D2) = (b*s + f) / e

			// If t in [0,1] done. Else clamp t, recompute s for the new value
			// of t using s = Dot((P2 + D2*t) - P1,D1) / Dot(D1,D1)= (t*b - c) / a
			// and clamp s to [0, 1]
			float tnom = b * s + f;
			if (tnom < 0.0f) {
				t = 0.0f;
				s = f_clamp(-c / a, 0.0f, 1.0f);
			}
			else if (tnom > e) {
				t = 1.0f;
				s = f_clamp((b - c) / a, 0.0f, 1.0f);
			}
			else {
				t = tnom / e;
			}
		}
	}
	c1 = f3_add(p1, f3_sclrmult(s, d1));
	c2 = f3_add(p2, f3_sclrmult(t, d2));
	float3 c2c1 = f3_sub(c1, c2);
	return f3_dot(c2c1, c2c1);
}

__device__ __host__ float point_to_triangle
	(float3 pt, float3& ptt, float3 a, float3 b, float3 c)
{
	// Check if P in vertex region outside A
	float3 ab = f3_sub(b, a);
	float3 ac = f3_sub(c, a);
	float3 ap = f3_sub(pt, a);
	float d1 = f3_dot(ab, ap);
	float d2 = f3_dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f) {
		ptt = a;
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (1,0,0)
	}
	// Check if P in vertex region outside B
	float3 bp = f3_sub(pt, b);
	float d3 = f3_dot(ab, bp);
	float d4 = f3_dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3) {
		ptt = b;
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (0,1,0)
	}
	// Check if P in edge region of AB, if so return projection of P onto AB
	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		ptt = f3_add(a, f3_sclrmult(v, ab));
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (1-v,v,0)
	}
	// Check if P in vertex region outside C
	float3 cp = f3_sub(pt, c);
	float d5 = f3_dot(ab, cp);
	float d6 = f3_dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6) {// barycentric coordinates (0,0,1)
		ptt = c;
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist);
	}
	// Check if P in edge region of AC, if so return projection of P onto AC
	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float w = d2 / (d2 - d6);
		ptt = f3_add(a, f3_sclrmult(w, ac));
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (1-w,0,w)
	}
	// Check if P in edge region of BC, if so return projection of P onto BC
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		ptt = f3_add(b, f3_sclrmult(w, f3_sub(c, b)));
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (0,1-w,w)
	}
	// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	// = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
	ptt = f3_add(a, f3_add(f3_sclrmult(v, ab), f3_sclrmult(w, ac)));
	float3 dist = f3_sub(pt, ptt);
	return f3_dot(dist, dist);
}

//, float3 &pr1, float3 &pr2) pr1,pr2 is nearest pair of points
__device__ __host__ bool IntersectTriangleTriangle(
	const float3 a1, const float3 b1, const float3 c1,
	const float3 a2, const float3 b2, const float3 c2
	const Triangle&)
{
	float3 pr1, pr2;
	float local_min;
	float temp;
	float s = 0, t = 0;
	float3 p1;
	float3 p2;
	local_min = edge_to_edge(a1, b1, a2, b2, s, t, p1, p2);
	pr1 = p1;
	pr2 = p2;
	temp = edge_to_edge(a1, c1, a2, b2, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c1, b1, a2, b2, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a1, b1, a2, c2, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a1, c1, a2, c2, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c1, b1, a2, c2, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a1, b1, c2, b2, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a1, c1, c2, b2, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c1, b1, c2, b2, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = a1;
	temp = point_to_triangle(p1, p2, a2, b2, c2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = b1;
	temp = point_to_triangle(p1, p2, a2, b2, c2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = c1;
	temp = point_to_triangle(p1, p2, a2, b2, c2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = a2;
	temp = point_to_triangle(p2, p1, a1, b1, c1);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = b2;
	temp = point_to_triangle(p2, p1, a1, b1, c1);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = c2;
	temp = point_to_triangle(p2, p1, a1, b1, c1);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	float3 dist = f3_sub(pr1, pr2);
	return (f3_dot(dist, dist) < EPSILON);
}

__device__ bool intersect(const BBox& bbox2, const Triangle& tri2, int* outHit,
	InternalNode* internalNodes, LeafNode* leafNodes, Triangle* myMesh) {
	bool HIT = false;
	int visit_stack[64] = { 0 };
	int level_stack[64] = { 0 };
	int stack_ptr = 1;
	while (stack_ptr > 0) {
		--stack_ptr;
		int idx = visit_stack[stack_ptr];
		int level = level_stack[stack_ptr];
		BBox box = internalNodes[idx].getBBox();
		bool bbox_hit = IntersectAABBAABB(bbox2, box);
		if (!bbox_hit) {
			// cout << "not hit at level " << level << endl;
			continue;
		} else {
			// cout << "hit at level " << level << endl;
		}
		///
		if (internalNodes[idx].rightNodeType() == LEAFNODE) {
			// this is incorrect
			// const float3& a = myMesh[internalNodes[idx].rightNodeIdx()].a;
			// const float3& b = myMesh[internalNodes[idx].rightNodeIdx()].b;
			// const float3& c = myMesh[internalNodes[idx].rightNodeIdx()].c;
			// corrected
			const Triangle& tri =
				myMesh[leafNodes[internalNodes[idx].rightNodeIdx()].getObjectID()];
#if 0 // debugging
			BBox boxleft = leafNodes
				[internalNodes[idx].rightNodeIdx()].getBBox();
			if (boxleft.Contains(a) && boxleft.Contains(b) && boxleft.Contains(c)) {
				cout << "contains" << endl;
			}
			else {
				cout << "not contains" << endl;
				cout << boxleft.toString();
				cout << a.x << "," << a.y << "," << a.z << endl;
				cout << b.x << "," << b.y << "," << b.z << endl;
				cout << c.x << "," << c.y << "," << c.z << endl;
				cout << level << endl;
			}
#endif
			// cout << "ray: " << "(";
			// cout << ray_orig.x << "," << ray_orig.y << "," << ray_orig.z;
			// cout << ") " << "(";
			// cout << ray_dir.x << "," << ray_dir.y << "," << ray_dir.z;
			// cout << ")" << endl;
			bool hit = tri2.Intersect(tri);
			// cout << "testing right leaf: " << near_idx << ", " << hit << endl;
			if (hit) {
				outHit[internalNodes[idx].rightNodeIdx()] = 1;
				HIT = true;
			}
		}
		else {
			visit_stack[stack_ptr] = internalNodes[idx].rightNodeIdx();
			level_stack[stack_ptr] = level + 1;
			++stack_ptr;
		}
		///
		if (internalNodes[idx].leftNodeType() == LEAFNODE) {
			// this is incorrect
			// const float3& a = myMesh[internalNodes[idx].leftNodeIdx()].a;
			// const float3& b = myMesh[internalNodes[idx].leftNodeIdx()].b;
			// const float3& c = myMesh[internalNodes[idx].leftNodeIdx()].c;
			// corrected
			const Triangle& tri =
				myMesh[leafNodes[internalNodes[idx].leftNodeIdx()].getObjectID()];
#if 0 // debugging
			BBox boxleft = leafNodes
				[internalNodes[idx].leftNodeIdx()].getBBox();
			if (boxleft.Contains(a) && boxleft.Contains(b) && boxleft.Contains(c)) {
				cout << "contains" << endl;
			}
			else {
				cout << "not contains" << endl;
				cout << boxleft.toString();
				cout << a.x << "," << a.y << "," << a.z << endl;
				cout << b.x << "," << b.y << "," << b.z << endl;
				cout << c.x << "," << c.y << "," << c.z << endl;
				cout << level << endl;
			}
#endif
			// cout << "ray: " << "(";
			// cout << ray_orig.x << "," << ray_orig.y << "," << ray_orig.z;
			// cout << ") " << "(";
			// cout << ray_dir.x << "," << ray_dir.y << "," << ray_dir.z;
			// cout << ")" << endl;
			bool hit = tri2.Intersect(tri);
			// cout << "testing left leaf: " << near_idx << ", " << hit << endl;
			if (hit) {
				outHit[internalNodes[idx].leftNodeIdx()] = 1;
				HIT = true;
			}
		}
		else {
			visit_stack[stack_ptr] = internalNodes[idx].leftNodeIdx();
			level_stack[stack_ptr] = level + 1;
			++stack_ptr;
		}
	}
	return HIT; // ray epsilon to mitigate self intersection
}

__global__ void intersectKernel(int total_boxes2,
	BBox* bboxes2, Triangle* tris2, int* outHit, int* outHit2,
	InternalNode* internalNodes, LeafNode* leafNodes, Triangle* myMesh)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= total_boxes2)
		return;
	if (intersect(bboxes2[idx], tris2[idx], outHit,
		internalNodes, leafNodes, myMesh))
	{
		outHit2[idx] = 1;
	}
	else {
		outHit2[idx] = 0;
	}
}

void CudaBVH::generateValues(int* keys) {
	int* d_keys;
	int size = SAMPLE_SIZE * sizeof(HashType);
	checkCudaErrors(cudaMalloc((void**)&d_keys, size));
	valuesKernel << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
		THREADS_PER_BLOCK >> > (SAMPLE_SIZE, d_keys);
	getLastCudaError("value err");
	checkCudaErrors(cudaMemcpy(keys, d_keys, size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_keys));
}

BVHTree CudaBVH::generateBVHTree(int* values,//element index in the unsorted array
	BBox* objects, Triangle* tris) {
	//for (int n = 0 ; n < SAMPLE_SIZE; n++) {
	//    BBox b = objects[n];
	//    Triangle t = tris[n];
	//    if (b.Contains(t.a) && b.Contains(t.b) && b.Contains(t.c)) {
	//        cout << "contains" << endl;
	//    }
	//    else{
	//        cout << "not contains" << endl;

	//        cout << b.toString();
	//        cout << t.a.x << "," << t.a.y << "," << t.a.z << endl;
	//        cout << t.b.x << "," << t.b.y << "," << t.b.z << endl;
	//        cout << t.c.x << "," << t.c.y << "," << t.c.z << endl;
	//    }
	//}

	////the unsorted element index
	//for (int n = 0; n < SAMPLE_SIZE; n++) {
	//    cout << values[n] << endl;
	//}

	myMesh.resize(SAMPLE_SIZE);
	for (int n = 0; n < SAMPLE_SIZE; n++) {
		myMesh[n] = tris[n];
	}
	myBBox.resize(SAMPLE_SIZE);
	for (int n = 0; n < SAMPLE_SIZE; n++) {
		myBBox[n] = objects[n];
	}
	checkCudaErrors(cudaMalloc((void**)&d_myMesh, SAMPLE_SIZE * sizeof(Triangle)));
	checkCudaErrors(cudaMemcpy(d_myMesh, &myMesh[0],
		SAMPLE_SIZE * sizeof(Triangle), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_myBBox, SAMPLE_SIZE * sizeof(BBox)));
	checkCudaErrors(cudaMemcpy(d_myBBox, &myBBox[0],
		SAMPLE_SIZE * sizeof(BBox), cudaMemcpyHostToDevice));
	HashType* d_objKeys;
	int* d_objValues;
	BBox* d_objects;
	cudaMalloc((void**)&d_objValues, SAMPLE_SIZE * sizeof(int));
	cudaMalloc((void**)&d_objKeys, SAMPLE_SIZE * sizeof(HashType));
	cudaMalloc((void**)&d_objects, SAMPLE_SIZE * sizeof(BBox));
	cudaMemcpy(d_objects, objects, SAMPLE_SIZE * sizeof(BBox),
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_objValues, values, SAMPLE_SIZE * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMallocManaged((void**)&mor, SAMPLE_SIZE * sizeof(MortonRec));
	cudaDeviceSynchronize();
	cudaEventRecord(start);
	morton3DCuda << <(SAMPLE_SIZE+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,
		THREADS_PER_BLOCK >> > (SAMPLE_SIZE, d_objKeys, d_objects, mor);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	/*{
		HashType *keys = new HashType[SAMPLE_SIZE];
		BBox *boxes = new BBox[SAMPLE_SIZE];
		checkCudaErrors(cudaMemcpy(keys, d_objKeys, sizeof(HashType)*SAMPLE_SIZE,
			cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(boxes, d_objects, sizeof(BBox) * SAMPLE_SIZE,
			cudaMemcpyDeviceToHost));
		for (int n = 0; n < SAMPLE_SIZE; n++) {
			cout << "key " << n << " = " << keys[n] << endl;
		}

		for (;;);
	}*/
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("It took me %f milliseconds to generate morton codes.\n", milliseconds);
	/*for (int i = 0; i < SAMPLE_SIZE; i++)
	printf("%d\n", d_keys[i]);*/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Radix sort
	int num_items = SAMPLE_SIZE;
	int size = SAMPLE_SIZE * sizeof(HashType);
	DoubleBuffer<HashType> d_sortedKeys;
	DoubleBuffer<int> d_sortedValues;
	CubDebugExit(g_allocator.DeviceAllocate(
		(void**)&d_sortedKeys.d_buffers[0], sizeof(HashType) * num_items));
	CubDebugExit(g_allocator.DeviceAllocate(
		(void**)&d_sortedKeys.d_buffers[1], sizeof(HashType) * num_items));
	CubDebugExit(g_allocator.DeviceAllocate(
		(void**)&d_sortedValues.d_buffers[0], sizeof(int) * num_items));
	CubDebugExit(g_allocator.DeviceAllocate(
		(void**)&d_sortedValues.d_buffers[1], sizeof(int) * num_items));

	// Allocate temporary storage
	size_t  temp_storage_bytes = 0;
	void* d_temp_storage = NULL;
	cudaEventRecord(start);
	CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage,
		temp_storage_bytes, d_sortedKeys, d_sortedValues, num_items));
	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	CubDebugExit(cudaMemcpy(d_sortedKeys.d_buffers[d_sortedKeys.selector],
		d_objKeys, sizeof(HashType) * num_items, cudaMemcpyDeviceToDevice));
	CubDebugExit(cudaMemcpy(d_sortedValues.d_buffers[d_sortedValues.selector],
		d_objValues, sizeof(int) * num_items, cudaMemcpyDeviceToDevice));
	CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
		d_sortedKeys, d_sortedValues, num_items));
	cudaEventRecord(stop);
	checkCudaErrors(cudaMemcpy(d_objValues, d_sortedValues.Current(),
		SAMPLE_SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_objKeys, d_sortedKeys.Current(),
		SAMPLE_SIZE * sizeof(HashType), cudaMemcpyDeviceToDevice));
	//{
	//	HashType* keys = new HashType[SAMPLE_SIZE];
	//	int* values = new int[SAMPLE_SIZE];
	//	checkCudaErrors(cudaMemcpy(keys, d_objKeys,
	//		sizeof(HashType) * SAMPLE_SIZE, cudaMemcpyDeviceToHost));
	//	checkCudaErrors(cudaMemcpy(values, d_objValues,
	//		sizeof(int) * SAMPLE_SIZE, cudaMemcpyDeviceToHost));
	//	for (int n = 0; n < SAMPLE_SIZE; n++) {
	//		cout << "key, value " << n << " = ";
	//		cout << keys[n] << ", " << values[n] << endl;
	//	}
	//	for (;;);
	//}
	//(FALSE)also apply the sorted order to the triangle array
	//{
	//	// the sorted element index
	//	cudaMemcpy(values, d_objValues, SAMPLE_SIZE * sizeof(int),
	//		cudaMemcpyDeviceToHost);
	//	//for (int n = 0; n < SAMPLE_SIZE; n++) {
	//	//    cout << values[n] << endl;
	//	//}

	//	vector<Triangle> oldMesh = myMesh;
	//	for (int n = 0; n < SAMPLE_SIZE; n++) {
	//		myMesh[n] = oldMesh[values[n]];
	//	}
	//}
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("It took me %f milliseconds to run parallel radix sort.\n", milliseconds);
	///debug
	/*int* vbuf = new int[SAMPLE_SIZE];
	HashType* kbuf = new HashType[SAMPLE_SIZE];
	cudaMemcpy(vbuf, d_objValues, SAMPLE_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(kbuf, d_objKeys, SAMPLE_SIZE*sizeof(HashType), cudaMemcpyDeviceToHost);
	for (int i = 0; i < SAMPLE_SIZE; i++)
	cout << "Sorted value: " << vbuf[i] << " key: " << bitset<64>(kbuf[i]) << "\n";
	*/
	/////Generate hierachy
	LeafNode* d_leafNodes;
	InternalNode* d_internalNodes;
	// Construct leaf nodes.
	// Note: This step can be avoided by storing
	// the tree in a slightly different way.
	cudaMalloc((void**)&d_leafNodes, SAMPLE_SIZE * sizeof(LeafNode));
	cudaMallocManaged(
		(void**)&d_internalNodes, (SAMPLE_SIZE - 1) * sizeof(InternalNode));
	for (int n = 0; n < SAMPLE_SIZE; n++) {
		d_internalNodes[n] = InternalNode();
	}
	checkCudaErrors(cudaDeviceSynchronize());
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	assignLeafNodes << <(SAMPLE_SIZE+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,
		THREADS_PER_BLOCK >> > (SAMPLE_SIZE, d_leafNodes, d_objValues, d_objects);
	//{
	//	// the sorted element index
	//	cudaMemcpy(values, d_objValues, SAMPLE_SIZE * sizeof(int),
	//		cudaMemcpyDeviceToHost);
	//	cudaMemcpy(objects, d_objects, SAMPLE_SIZE * sizeof(int),
	//		cudaMemcpyDeviceToHost);
	//	//             for (int n = 0; n < SAMPLE_SIZE; n++) {
	//	//                 cout << values[n] << endl;
	//	//             }

	//	for (int n = 0; n < SAMPLE_SIZE; n++) {
	//		BBox b = objects[values[n]];
	//		Triangle t = myMesh[values[n]];
	//		if (b.Contains(t.a) && b.Contains(t.b) && b.Contains(t.c)) {
	//			//                     cout << "contains" << endl;
	//		}
	//		else {
	//			cout << "not contains" << endl;

	//			cout << b.toString();
	//			cout << t.a.x << "," << t.a.y << "," << t.a.z << endl;
	//			cout << t.b.x << "," << t.b.y << "," << t.b.z << endl;
	//			cout << t.c.x << "," << t.c.y << "," << t.c.z << endl;
	//		}
	//	}
	//	for (;;);
	//}
	assignInternalNodes 
		<<<(SAMPLE_SIZE+THREADS_PER_BLOCK-2)/THREADS_PER_BLOCK,
		THREADS_PER_BLOCK >>>
		(SAMPLE_SIZE, d_objKeys, d_leafNodes, d_internalNodes, d_objValues);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("It took me %f milliseconds to generate hierachy.\n", milliseconds);
	cudaFree(d_objKeys); cudaFree(d_objValues); cudaFree(d_objects);
	/*LeafNode* leafNodes = new LeafNode[SAMPLE_SIZE];
	InternalNode* internalNodes = new InternalNode[SAMPLE_SIZE - 1];
	cudaMemcpy(leafNodes, d_leafNodes, SAMPLE_SIZE*sizeof(LeafNode),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(internalNodes, d_internalNodes,
		(SAMPLE_SIZE-1)*sizeof(InternalNode), cudaMemcpyDeviceToHost);
	for (int i = 0; i < SAMPLE_SIZE; ++i)
	cout << internalNodes[i].getIdx() << " " << internalNodes[i].getParent() << "\n";*/
	/////Assign bounding box to internal nodes
	int* atom;
	cudaMalloc((void**)&atom, SAMPLE_SIZE * sizeof(int));
	cudaMemset(atom, 0, SAMPLE_SIZE * sizeof(int));
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	internalNodeBBox << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
		THREADS_PER_BLOCK >> >
		(SAMPLE_SIZE, atom, d_internalNodes, d_leafNodes, d_myBBox);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("It took me %f milliseconds to assign bounding box.\n", milliseconds);
	LeafNode* leafNodes = new LeafNode[SAMPLE_SIZE];
	InternalNode* internalNodes = new InternalNode[SAMPLE_SIZE - 1];
	cudaMemcpy(leafNodes, d_leafNodes,
		SAMPLE_SIZE * sizeof(LeafNode), cudaMemcpyDeviceToHost);
	cudaMemcpy(internalNodes, d_internalNodes,
		(SAMPLE_SIZE - 1) * sizeof(InternalNode), cudaMemcpyDeviceToHost);
	BVHTree buf;
	buf.internalNodes = internalNodes;
	buf.leafNodes = leafNodes;
	myTree = buf;
	BVHTree d_buf;
	d_buf.internalNodes = d_internalNodes;
	d_buf.leafNodes = d_leafNodes;
	d_myTree = d_buf;
	//printBVH(internalNodes, leafNodes);
	//cudaFree(d_internalNodes); cudaFree(d_leafNodes);
}

CudaBVH::~CudaBVH() {
	checkCudaErrors(cudaFree(d_myTree.internalNodes));
	checkCudaErrors(cudaFree(d_myTree.leafNodes));
	checkCudaErrors(cudaFree(d_myMesh));
	checkCudaErrors(cudaFree(d_myBBox));
	cudaFree(mor);
}

// 	CudaBVH::CudaBVH() {
// 		SAMPLE_SIZE = 10;
// 		THREADS_PER_BLOCK = 5;
// 		BBox* dummy = new BBox[SAMPLE_SIZE];
// 		generateSampleDataset(dummy);
// 		CudaBVH(dummy, SAMPLE_SIZE, THREADS_PER_BLOCK);
// 		free(dummy);
// 	}

// 	CudaBVH(int sample_size, int threads_per_block) {
// 		SAMPLE_SIZE = sample_size;
// 		THREADS_PER_BLOCK = threads_per_block;
// 		BBox* dummy = new BBox[SAMPLE_SIZE];
// 		generateSampleDataset(dummy);
// 		Init(dummy, SAMPLE_SIZE, THREADS_PER_BLOCK);
//         delete[] dummy;
// 	}

CudaBVH::CudaBVH(BBox* objects, Triangle* tris, int sample_size,
	int threads_per_block)
{
	SAMPLE_SIZE = sample_size;
	THREADS_PER_BLOCK = threads_per_block;
	Init(objects, tris, SAMPLE_SIZE, THREADS_PER_BLOCK);
}

void CudaBVH::Init(BBox* objects, Triangle* tris, int sample_size,
	int threads_per_block)
{
	SAMPLE_SIZE = sample_size;
	THREADS_PER_BLOCK = threads_per_block;
	cout << SAMPLE_SIZE << " " << THREADS_PER_BLOCK << endl;
	int* values;
	values = new int[SAMPLE_SIZE];
	// 		generateValues(values);
	//         for (int n = 0; n < SAMPLE_SIZE; n++) {
	//             cout << "dfhgsd" << values[n] << endl;
	//         }
	for (int n = 0; n < SAMPLE_SIZE; n++) {
		values[n] = n;
	}
	generateBVHTree(values, objects, tris);
	delete[] values;
}

void CudaBVH::generateSampleDataset(BBox* objects) {
	float buf[6];
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		for (int j = 0; j < 6; j++)
			buf[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		objects[i]._max.x = max(buf[0], buf[1]);
		objects[i]._min.x = min(buf[0], buf[1]);
		objects[i]._max.y = max(buf[2], buf[3]);
		objects[i]._min.y = min(buf[2], buf[3]);
		objects[i]._max.z = max(buf[4], buf[5]);
		objects[i]._min.z = min(buf[4], buf[5]);
	}
}

#if 0
void CudaBVH::printBVH(int idx, int level) {
	cout << "Internal (" << level << ") " << idx << " ";
	cout << myTree.internalNodes[idx].getBBox().toString() << "\n";
	if (myTree.internalNodes[idx].leftNodeType() == LEAFNODE) {
		cout << "Leaf (l) " << myTree.leafNodes
			[myTree.internalNodes[idx].leftNodeIdx()].getObjectID() << " "
			<< myTree.leafNodes
			[myTree.internalNodes[idx].leftNodeIdx()].getBBox().toString()
			<< "\n";
	} else
		printBVH(myTree.internalNodes[idx].leftNodeIdx(), level + 1);

	if (myTree.internalNodes[idx].rightNodeType() == LEAFNODE) {
		cout << "Leaf (r) " << myTree.leafNodes
			[myTree.internalNodes[idx].rightNodeIdx()].getObjectID() << " "
			<< myTree.leafNodes
			[myTree.internalNodes[idx].rightNodeIdx()].getBBox().toString()
			<< "\n";
	} else printBVH(myTree.internalNodes[idx].rightNodeIdx(), level + 1);
}
#else
void CudaBVH::printBVH(InternalNode* internalNodes, LeafNode* leafNodes) {
	int visit_stack[64] = { 0 };
	int level_stack[64] = { 0 };
	int stack_ptr = 1;
	while (stack_ptr > 0) {
		--stack_ptr;
		int idx = visit_stack[stack_ptr];
		int level = level_stack[stack_ptr];
		//int pid = myTree.internalNodes[idx].getParent();
		//cout << idx << ", " << level << ", " << stack_ptr << endl;
		//if (pid == -1) {
		//	cout << "Internal (" << level << ") " << idx << " "
		//		<< myTree.internalNodes[idx].getBBox().toString() << endl;
		//}
		if (level > 100 /*idx > 38000*/) {
			cout << "===============================" << endl;
			cout << "node info {" << endl;
			cout << "  level: " << level << endl;
			cout << "  node idx: " << idx << ", "
				<< myTree.internalNodes[idx].getIdx() << endl;
			cout << "  node type: "
				<< myTree.internalNodes[idx].getType() << endl;
			cout << "  node.parent idx: "
				<< myTree.internalNodes[idx].getParent() << endl;
			cout << "  ndoe.parent type: "
				<< myTree.internalNodes[idx].parentType() << endl;
			cout << "  node.left idx: "
				<< myTree.internalNodes[idx].leftNodeIdx() << endl;
			cout << "  node.left type: "
				<< myTree.internalNodes[idx].leftNodeType() << endl;
			cout << "  node.right idx: "
				<< myTree.internalNodes[idx].rightNodeIdx() << endl;
			cout << "  node.right type: "
				<< myTree.internalNodes[idx].rightNodeType() << endl;
			cout << "}" << endl;
		}
		if (myTree.internalNodes[idx].rightNodeType() == LEAFNODE) {
			//if (myTree.leafNodes
			//	[myTree.internalNodes[idx].rightNodeIdx()].getParent() == -1)
			//{
			//	cout << "Leaf (r) " << myTree.leafNodes
			//		[myTree.internalNodes[idx].rightNodeIdx()].getObjectID()
			//		<< " " << myTree.leafNodes
			//		[myTree.internalNodes[idx].rightNodeIdx()].getBBox().toString()
			//		<< endl;
			//}
		} else {
			visit_stack[stack_ptr] = myTree.internalNodes[idx].rightNodeIdx();
			level_stack[stack_ptr] = level + 1;
			stack_ptr++;
		}
		if (myTree.internalNodes[idx].leftNodeType() == LEAFNODE) {
			//if (myTree.leafNodes
			//	[myTree.internalNodes[idx].leftNodeIdx()].getParent() == -1)
			//{
			//	cout << "Leaf (l) " << myTree.leafNodes
			//		[myTree.internalNodes[idx].leftNodeIdx()].getObjectID()
			//		<< " " << myTree.leafNodes
			//		[myTree.internalNodes[idx].leftNodeIdx()].getBBox().toString()
			//		<< endl;
			//}
		} else {
			visit_stack[stack_ptr] = myTree.internalNodes[idx].leftNodeIdx();
			level_stack[stack_ptr] = level + 1;
			stack_ptr++;
		}
	}
}
#endif

#if 0
void CudaBVH::drawBVHRecursive(int idx, int level) {
	if (level > 32) return;
	float color[3] = { level * 0.03, 1 - level * 0.03, 0 };
	myTree.internalNodes[idx].getBBox().Draw(color);
	if (myTree.internalNodes[idx].leftNodeType() == LEAFNODE) {
		myTree.leafNodes
			[myTree.internalNodes[idx].leftNodeIdx()].getBBox().Draw(color);
	} else
		drawBVHRecursive(myTree.internalNodes[idx].leftNodeIdx(), level + 1);
	if (myTree.internalNodes[idx].rightNodeType() == LEAFNODE) {
		myTree.leafNodes
			[myTree.internalNodes[idx].rightNodeIdx()].getBBox().Draw(color);
	} else drawBVHRecursive(myTree.internalNodes[idx].rightNodeIdx(), level+1);
}
void CudaBVH::draw() {
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	drawBVHRecursive(0, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
#else
void CudaBVH::draw(int levelDisplay) {
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	int visit_stack[64] = { 0 };
	int level_stack[64] = { 0 };
	int stack_ptr = 1;
	while (stack_ptr > 0) {
		--stack_ptr;
		int idx = visit_stack[stack_ptr];
		int level = level_stack[stack_ptr];
		//if (level > 32) return;
		float color[3] = { level * 0.03, 1 - level * 0.03, 0 };
		if (level == levelDisplay)
			myTree.internalNodes[idx].getBBox().Draw(color);
		if (myTree.internalNodes[idx].rightNodeType() == LEAFNODE) {
			//myTree.leafNodes
			//[myTree.internalNodes[idx].rightNodeIdx()].getBBox().Draw(color);
		}
		else {
			visit_stack[stack_ptr] = myTree.internalNodes[idx].rightNodeIdx();
			level_stack[stack_ptr] = level + 1;
			stack_ptr++;
		}
		if (myTree.internalNodes[idx].leftNodeType() == LEAFNODE) {
			//myTree.leafNodes
			//[myTree.internalNodes[idx].leftNodeIdx()].getBBox().Draw(color);
		}
		else {
			visit_stack[stack_ptr] = myTree.internalNodes[idx].leftNodeIdx();
			level_stack[stack_ptr] = level + 1;
			stack_ptr++;
		}
	}
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
#endif

void CudaBVH::drawTriangles() {
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	int visit_stack[64] = { 0 };
	int level_stack[64] = { 0 };
	int stack_ptr = 1;
	while (stack_ptr > 0) {
		--stack_ptr;
		int idx = visit_stack[stack_ptr];
		int level = level_stack[stack_ptr];
		if (level > 32) return;
		//float color[3] = { level * 0.03, 1 - level * 0.03, 0 };
		if (myTree.internalNodes[idx].rightNodeType() == LEAFNODE) {
			myMesh[myTree.internalNodes[idx].rightNodeIdx()].Draw();//color);
		}
		else {
			visit_stack[stack_ptr] = myTree.internalNodes[idx].rightNodeIdx();
			level_stack[stack_ptr] = level + 1;
			stack_ptr++;
		}
		if (myTree.internalNodes[idx].leftNodeType() == LEAFNODE) {
			myMesh[myTree.internalNodes[idx].leftNodeIdx()].Draw();// color);
		}
		else {
			visit_stack[stack_ptr] = myTree.internalNodes[idx].leftNodeIdx();
			level_stack[stack_ptr] = level + 1;
			stack_ptr++;
		}
	}
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void CudaBVH::drawTrianglesDEBUG() {
	//         glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	int visit_stack[64] = { 0 };
	int level_stack[64] = { 0 };
	int stack_ptr = 1;
	while (stack_ptr > 0) {
		--stack_ptr;
		int idx = visit_stack[stack_ptr];
		int level = level_stack[stack_ptr];
		if (level > 64) return;
		//float color[3] = { level * 0.03, 1 - level * 0.03, 0 };
		//myTree.internalNodes[idx].getBBox().Draw(color);
		if (myTree.internalNodes[idx].rightNodeType() == LEAFNODE) {
			// also works with large mesh
			//myMesh[myTree.internalNodes[idx].rightNodeIdx()].draw(color);
			//myMesh[myTree.internalNodes[idx].rightNodeIdx()].
			//getBBox().Draw(color);
			//myBBox[myTree.internalNodes[idx].rightNodeIdx()].Draw(color);
			//myTree.leafNodes
			//[myTree.internalNodes[idx].rightNodeIdx()].bbox_debug.Draw(color);//=
			//myTree.leafNodes
			//[myTree.internalNodes[idx].rightNodeIdx()].getBBox().Draw(color);//-
			myMesh[myTree.leafNodes
				[myTree.internalNodes[idx].rightNodeIdx()].
				getObjectID()].Draw();// color);//- only works with small mesh
		}
		else {
			visit_stack[stack_ptr] = myTree.internalNodes[idx].rightNodeIdx();
			level_stack[stack_ptr] = level + 1;
			stack_ptr++;
		}
		if (myTree.internalNodes[idx].leftNodeType() == LEAFNODE) {
			// also works with large mesh
			//myMesh[myTree.internalNodes[idx].leftNodeIdx()].Draw(color);
			//myMesh[myTree.internalNodes[idx].leftNodeIdx()].
			//getBBox().Draw(color);
			//myBBox[myTree.internalNodes[idx].leftNodeIdx()].Draw(color);
			//myTree.leafNodes[myTree.internalNodes[idx].leftNodeIdx()].
			//bbox_debug.Draw(color);//=
			//myTree.leafNodes[myTree.internalNodes[idx].leftNodeIdx()].
			//getBBox().Draw(color);//-
			myMesh[myTree.leafNodes[myTree.internalNodes[idx].leftNodeIdx()].
				getObjectID()].Draw();// color);//- only works with small mesh
		}
		else {
			visit_stack[stack_ptr] = myTree.internalNodes[idx].leftNodeIdx();
			level_stack[stack_ptr] = level + 1;
			stack_ptr++;
		}
	}
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void CudaBVH::boxIntersect(int boxes2_count, BBox* bboxes2, Triangle* tris2,
	int* outHit, int* outHit2)
{
	intersectKernel <<<(boxes2_count+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,
		THREADS_PER_BLOCK >>>
		(boxes2_count, bboxes2, tris2, outHit, outHit2,
			d_myTree.internalNodes, d_myTree.leafNodes, d_myMesh);
	////for debugging
	//for (int n = 0; n < ray_count; n++) {
	//	outHit[n] = intersect
	//		(ray_orig[n], ray_dir[n], outT[n], outU[n], outV[n], outIdx[n]);
	//}
}

//bool CudaBVH::intersect(const float3& ray_orig, const float3& ray_dir,
//	float& outT, float& outU, float& outV, int& outIdx)
//{
//	float3* d_rayorig;
//	float3* d_raydir;
//	float* d_t;
//	float* d_u;
//	float* d_v;
//	int* d_idx;
//	int* d_hit;
//	checkCudaErrors(cudaMallocManaged((void**)&d_rayorig, sizeof(float3)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_raydir, sizeof(float3)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_t, sizeof(float)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_u, sizeof(float)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_v, sizeof(float)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_idx, sizeof(int)));
//	checkCudaErrors(cudaMallocManaged((void**)&d_hit, sizeof(int)));
//	d_rayorig[0] = ray_orig;
//	d_raydir[0] = ray_dir;
//	checkCudaErrors(cudaDeviceSynchronize());
//	intersectKernel << <1, 1 >> > (1, d_rayorig, d_raydir, d_t, d_u, d_v,
//		d_idx, d_hit, d_myTree.internalNodes, d_myTree.leafNodes, d_myMesh);
//	checkCudaErrors(cudaDeviceSynchronize());
//	outT = d_t[0];
//	outU = d_u[0];
//	outV = d_v[0];
//	outIdx = d_idx[0];
//	int outHit = d_hit[0];
//	checkCudaErrors(cudaFree(d_rayorig));
//	checkCudaErrors(cudaFree(d_raydir));
//	checkCudaErrors(cudaFree(d_t));
//	checkCudaErrors(cudaFree(d_u));
//	checkCudaErrors(cudaFree(d_v));
//	checkCudaErrors(cudaFree(d_idx));
//	checkCudaErrors(cudaFree(d_hit));
//	return outHit;
//}