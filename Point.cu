#include <cuda_runtime.h>
#include "Point.cuh"
#include "BBox.cuh"

Point Point::operator-() const {
	return Point(-x, -y, -z);
}

Point Point::operator*(float a) const {
	return Point(x * a, y * a, z * a);
}

Point Point::operator/(float a) const {
	float temp = 1.0f / a;
	return Point(x * temp, y * temp, z * temp);
}

Point& Point::operator*=(float a) {
	x *= a;
	y *= a;
	z *= a;
	return *this;
}

Point& Point::operator/=(float a) {
	float temp = 1.0f / a;
	x *= temp;
	y *= temp;
	z *= temp;
	return *this;
}

Point Point::operator+(const Point& pt) const {
	return Point(x + pt.x, y + pt.y, z + pt.z);
}

Point& Point::operator+=(const Point& pt) {
	x += pt.x;
	y += pt.y;
	z += pt.z;
	return *this;
}

Point& Point::operator-=(const Point& pt) {
	x -= pt.x;
	y -= pt.y;
	z -= pt.z;
	return *this;
}

Point::operator float3() {
	return make_float3(x, y, z);
}