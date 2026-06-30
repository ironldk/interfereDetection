#ifndef BBOX_CUH_
#define BBOX_CUH_

#include "Point.cuh"
#include <math.h>
#include "Triangle.cuh"

#pragma pack(push, 1)
class BBox {
	friend class ObjLoader;
public:
	__host__ __device__ BBox(
		float x0 = FLT_MAX, float x1 = FLT_MIN,
		float y0 = FLT_MAX, float y1 = FLT_MIN,
		float z0 = FLT_MAX, float z1 = FLT_MIN
	): _min(x0, y0, z0), _max(x1, y1, z1) {}

	__host__ __device__ BBox(const Triangle& iT);
	__host__ __device__ BBox(const Point& iPt1, const Point& iPt2);

	__host__ __device__ BBox& Union(const BBox& b);
	__host__ __device__ BBox& Union(const Point& iPt);

	__host__ __device__ bool Intersect(const BBox& b) const;

	__host__ __device__ void Complete();

	__host__ __device__ void GetCenter(float& ox, float& oy, float& oz) const;
	__host__ __device__ void ComputeMin(float iMatrixModel[16], float oMin[3]) const;

#ifndef DEBUG
private:
#endif
	Point
		_min , _x100, _x001, _x101, _x010, _x110, _x011, _max ,
		_y000, _y010, _y001, _y011, _y100, _y110, _y101, _y111,
		_z000, _z001, _z010, _z011, _z100, _z101, _z110, _z111;
};
#pragma pack(pop)

// 静态断言验证BBox的内存布局
static_assert(sizeof(BBox) == 24 * 12, "BBox size must be 288 bytes (24 Points x 12 bytes)");

inline __host__ __device__ BBox::BBox(const Triangle& iT) : BBox(iT.a, iT.b) {
	Union(iT.c);
}

inline __host__ __device__ BBox::BBox(const Point& iPt1, const Point& iPt2) {
	float arr[3][2]{{iPt1.x, iPt2.x}, {iPt1.y, iPt2.y}, {iPt1.z, iPt2.z}};
	bool select[3]{iPt1.x < iPt2.x, iPt1.y < iPt2.y, iPt1.z < iPt2.z};
	_min.x = arr[0][1-select[0]];
	_max.x = arr[0][  select[0]];
	_min.y = arr[1][1-select[1]];
	_max.y = arr[1][  select[1]];
	_min.z = arr[2][1-select[2]];
	_max.z = arr[2][  select[2]];
}

inline __host__ __device__ BBox& BBox::Union(const BBox& b) {
	bool select[6]{
		_min.x > b._min.x, _min.y > b._min.y, _min.z > b._min.z,
		_max.x < b._max.x, _max.y < b._max.y, _max.z < b._max.z
	};
	float arr[6][2]{
		{_min.x, b._min.x}, {_min.y, b._min.y}, {_min.z, b._min.z},
		{_max.x, b._max.x}, {_max.y, b._max.y}, {_max.z, b._max.z}
	};
	_min.x = arr[0][select[0]];
	_min.y = arr[1][select[1]];
	_min.z = arr[2][select[2]];
	_max.x = arr[3][select[3]];
	_max.y = arr[4][select[4]];
	_max.z = arr[5][select[5]];
	return *this;
}

inline __host__ __device__ BBox& BBox::Union(const Point& iPt) {
	bool select[6]{
		_min.x > iPt.x, _min.y > iPt.y, _min.z > iPt.z,
		_max.x < iPt.x, _max.y < iPt.y, _max.z < iPt.z
	};
	float arr[6][2]{
		{_min.x, iPt.x}, {_min.y, iPt.y}, {_min.z, iPt.z},
		{_max.x, iPt.x}, {_max.y, iPt.y}, {_max.z, iPt.z}
	};
	_min.x = arr[0][select[0]];
	_min.y = arr[1][select[1]];
	_min.z = arr[2][select[2]];
	_max.x = arr[3][select[3]];
	_max.y = arr[4][select[4]];
	_max.z = arr[5][select[5]];
	return *this;
}

inline __host__ __device__ bool BBox::Intersect(const BBox& b) const {
	// Exit with no intersection if separated along an axis
	// Overlapping on all axes means AABBs are intersecting
	return !(
		(_max.x < b._min.x) + (_min.x > b._max.x) +
		(_max.y < b._min.y) + (_min.y > b._max.y) +
		(_max.z < b._min.z) + (_min.z > b._max.z)
	);
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

inline __host__ __device__ void BBox::ComputeMin(float iMatrixModel[16], float oMin[3]) const {
	Point arrBox[8]{ _min, _x001, _x010, _x011, _x100, _x101, _x110, _max };
	for (int i = 0; i < 8; ++i) {
		arrBox[i].Transform(iMatrixModel);
		float arr[3][2]{{arrBox[i].x, oMin[0]}, {arrBox[i].y, oMin[1]}, {arrBox[i].z, oMin[2]}};
		bool select[3]{arrBox[i].x > oMin[0], arrBox[i].y > oMin[1], arrBox[i].z > oMin[2]};
		oMin[0] = arr[0][select[0]];
		oMin[1] = arr[1][select[1]];
		oMin[2] = arr[2][select[2]];
	}
}
#endif