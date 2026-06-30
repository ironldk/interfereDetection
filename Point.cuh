#ifndef _POINT_H_
#define _POINT_H_

#include <stdio.h>
#include "vector_types.h"
#include <math.h>
class BBox;
class Triangle;

#define HASH_64 1
#if HASH_64
typedef unsigned __int64 HashType;
#else
typedef unsigned int HashType;
#endif

//#define DEBUG INT_MAX
#ifdef DEBUG
#define CUDA_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define CUDA_DEBUG_PRINT(...)
#endif


#pragma pack(push, 1)
class Point {
	friend class ObjLoader;
	friend class BBox;
	friend class Triangle;
	friend Point operator*(float k, const Point& pt);
	friend __host__ __device__ Point cross(const Point& p1, const Point& p2);
public:
	// constructors
	__host__ __device__ Point(float nx = 0.0, float ny = 0.0, float nz = 0.0)
		: x(nx), y(ny), z(nz) {};

	__host__ __device__ Point(const float3& pt)
		: x(pt.x), y(pt.y), z(pt.z) {};

	__host__ __device__ Point(const Point& pt)
		: x(pt.x), y(pt.y), z(pt.z) {};

	__host__ __device__ Point& operator=(const Point& pt);

	// overloaded operators
	Point operator-() const;
	__host__ __device__ Point operator*(float a) const;
	Point operator/(float a) const;
	Point& operator*=(float a);
	Point& operator/=(float a);

	__host__ __device__ Point operator+(const Point& pt) const;
	Point& operator+=(const Point& pt);
	Point& operator-=(const Point& pt);
	__host__ __device__ Point operator-(const Point& pt) const;
	__host__ __device__ float operator*(const Point& pt) const;

	operator float3();

	__host__ __device__ Point& Transform(float iMatrixModel[16]);

#ifndef DEBUG
private:
#endif
	float x, y, z;
};
#pragma pack(pop)

// 静态断言验证Point的内存布局
static_assert(sizeof(Point) == 12, "Point size must be 12 bytes (3 floats)");

// friend
inline Point operator*(float k, const Point &pt) {
	return Point(k*pt.x, k*pt.y, k*pt.z);
}

inline __host__ __device__ Point cross(const Point& p1, const Point& p2) {
	return Point(
		fmaf(p1.y, p2.z, -(p1.z * p2.y)),
		fmaf(p1.z, p2.x, -(p1.x * p2.z)),
		fmaf(p1.x, p2.y, -(p1.y * p2.x))
	);
}

inline __host__ __device__ Point& Point::operator=(const Point& pt) {
	x = pt.x;
	y = pt.y;
	z = pt.z;
	return *this;
}

inline __host__ __device__ Point Point::operator*(float a) const {
	return Point(x * a, y * a, z * a);
}

inline __host__ __device__ Point Point::operator+(const Point& pt) const {
	return Point(x + pt.x, y + pt.y, z + pt.z);
}


inline __host__ __device__ Point Point::operator-(const Point& pt) const {
	return Point(x - pt.x, y - pt.y, z - pt.z);
}

inline __host__ __device__ float Point::operator*(const Point& pt) const {
	return fmaf(x, pt.x, fmaf(y, pt.y, z * pt.z));
}

inline __host__ __device__ Point& Point::Transform(float iMatrixModel[16]){
	float X = fmaf(iMatrixModel[0], x, fmaf(iMatrixModel[4], y, fmaf(iMatrixModel[ 8], z, iMatrixModel[12])));
	float Y = fmaf(iMatrixModel[1], x, fmaf(iMatrixModel[5], y, fmaf(iMatrixModel[ 9], z, iMatrixModel[13])));
	float Z = fmaf(iMatrixModel[2], x, fmaf(iMatrixModel[6], y, fmaf(iMatrixModel[10], z, iMatrixModel[14])));
	x = X;
	y = Y;
	z = Z;
	return *this;
}

#endif
