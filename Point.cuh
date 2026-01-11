#ifndef _POINT_H_
#define _POINT_H_

#include "vector_types.h"
class BBox;

class Point {
	friend class ObjLoader;
	friend class Triangle;
	friend class BBox;
	friend Point operator*(float k, const Point& pt);
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
	Point operator*(float a) const;
	Point operator/(float a) const;
	Point& operator*=(float a);
	Point& operator/=(float a);
	Point operator+(const Point& pt) const;
	Point& operator+=(const Point& pt);
	Point& operator-=(const Point& pt);
	__host__ __device__ Point operator-(const Point& pt) const;
	__host__ __device__ float operator*(const Point& pt) const;

	operator float3();

	__host__ __device__ Point& Transform(float iMatrixModel[16]);
private:
	float x, y, z;
};

// friend
inline Point operator*(float k, const Point &pt) {
	return Point(k*pt.x, k*pt.y, k*pt.z);
}

inline __host__ __device__ Point& Point::operator=(const Point& pt) {
	x = pt.x;
	y = pt.y;
	z = pt.z;
	return *this;
}

inline Point Point::operator-(const Point& pt) const {
	return Point(x - pt.x, y - pt.y, z - pt.z);
}

inline __host__ __device__ float Point::operator*(const Point& pt) const {
	return x * pt.x + y * pt.y + z * pt.z;
}

inline __host__ __device__ Point& Point::Transform(float iMatrixModel[16]){
	x = iMatrixModel[0] * x + iMatrixModel[4] * y + iMatrixModel[ 8] * z + iMatrixModel[12];
	y = iMatrixModel[1] * x + iMatrixModel[5] * y + iMatrixModel[ 9] * z + iMatrixModel[13];
	z = iMatrixModel[2] * x + iMatrixModel[6] * y + iMatrixModel[10] * z + iMatrixModel[14];
	return *this;
}

#endif