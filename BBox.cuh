#ifndef BBOX_CUH_
#define BBOX_CUH_

#include "Point.cuh"
#include <math.h>
#include "Triangle.cuh"

class BBox {
	friend class ObjLoader;
public:
	__host__ __device__ BBox(
		float x0 = 0.0, float x1 = 0.0,
		float y0 = 0.0, float y1 = 0.0,
		float z0 = 0.0, float z1 = 0.0
	): _min(x0, y0, z0), _max(x1, y1, z1) {}

	__host__ __device__ BBox(const Triangle& iT);
	__host__ __device__ BBox(const Point& iPt1, const Point& iPt2);

	BBox& operator+=(const Point& pt);
	BBox& operator-=(const Point& pt);
	BBox& operator*=(const float& scale);

	__host__ __device__ BBox& Union(const BBox& b);
	__host__ __device__ BBox& Union(const Point& iPt);

	__host__ __device__ bool Intersect(const BBox& b) const;

	float LargestEdge();
	const Point& GetMin();
	__host__ __device__ void Complete();

	__host__ __device__ void GetCenter(float& ox, float& oy, float& oz) const;

private:
	Point
		_min , _x100, _x001, _x101, _x010, _x110, _x011, _max ,
		_y000, _y010, _y001, _y011, _y100, _y110, _y101, _y111,
		_z000, _z001, _z010, _z011, _z100, _z101, _z110, _z111;
};

inline __host__ __device__ BBox::BBox(const Triangle& iT) : BBox(iT.a, iT.b) {
	Union(iT.c);
}

inline __host__ __device__ BBox::BBox(const Point& iPt1, const Point& iPt2) {
	if (iPt1.x < iPt2.x) { _min.x = iPt1.x; _max.x = iPt2.x; } else { _max.x = iPt1.x; _min.x = iPt2.x; }
	if (iPt1.y < iPt2.y) { _min.y = iPt1.y; _max.y = iPt2.y; } else { _max.y = iPt1.y; _min.y = iPt2.y; }
	if (iPt1.z < iPt2.z) { _min.z = iPt1.z; _max.z = iPt2.z; } else { _max.z = iPt1.z; _min.z = iPt2.z; }
}

inline __host__ __device__ BBox& BBox::Union(const BBox& b) {
	if (_min.x > b._min.x) { _min.x = b._min.x; }
	if (_max.x < b._max.x) { _max.x = b._max.x; }
	if (_min.y > b._min.y) { _min.y = b._min.y; }
	if (_max.y < b._max.y) { _max.y = b._max.y; }
	if (_min.z > b._min.z) { _min.z = b._min.z; }
	if (_max.z < b._max.z) { _max.z = b._max.z; }
	return *this;
}

inline __host__ __device__ BBox& BBox::Union(const Point& iPt) {
	     if (_min.x > iPt.x) { _min.x = iPt.x; }
	else if (_max.x < iPt.x) { _max.x = iPt.x; }
	     if (_min.y > iPt.y) { _min.y = iPt.y; }
	else if (_max.y < iPt.y) { _max.y = iPt.y; }
	     if (_min.z > iPt.z) { _min.z = iPt.z; }
	else if (_max.z < iPt.z) { _max.z = iPt.z; }
	return *this;
}

inline __host__ __device__ bool BBox::Intersect(const BBox& b) const {
	// Exit with no intersection if separated along an axis
	if (_max.x < b._min.x || _min.x > b._max.x) {return false;}
	if (_max.y < b._min.y || _min.y > b._max.y) {return false;}
	if (_max.z < b._min.z || _min.z > b._max.z) {return false;}
	// Overlapping on all axes means AABBs are intersecting
	return true;
}

inline __host__ __device__ void BBox::GetCenter(float& ox, float& oy, float& oz) const {
	ox = (_min.x + _max.x) * 0.5f;
	oy = (_min.y + _max.y) * 0.5f;
	oz = (_min.z + _max.z) * 0.5f;
}

inline __host__ __device__ void BBox::Complete() {
	_z000 = _y000 = _min;
	_z001 = _y001 = _x001 = Point(_min.x, _min.y, _max.z);
	_z010 = _y010 = _x010 = Point(_min.x, _max.y, _min.z);
	_z011 = _y011 = _x011 = Point(_min.x, _max.y, _max.z);
	_z100 = _y100 = _x100 = Point(_max.x, _min.y, _min.z);
	_z101 = _y101 = _x101 = Point(_max.x, _min.y, _max.z);
	_z110 = _y110 = _x110 = Point(_max.x, _max.y, _min.z);
	_z111 = _y111 = _max;
}

#endif