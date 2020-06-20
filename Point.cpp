#include "Point.h"
Point::Point(const Point& pt):
	x(pt.x), y(pt.y), z(pt.z) {}
Point::Point(float nx, float ny, float nz):
	x(nx), y(ny), z(nz) {}

void Point::read(istream& is) {
	is >> x;
	is >> y;
	is >> z;
}
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
Point Point::operator-(const Point& pt) const {
	return Point(x - pt.x, y - pt.y, z - pt.z);
}
float Point::operator*(const Point& pt) const {
	return x*pt.x + y*pt.y + z*pt.z;
}
Point::operator float3() {
	return make_float3(x, y, z);
}
Point::operator BBox() const {
	return BBox(x, x, y, y, z, z);
}

// |vtx.x|   |mat[0] mat[1] mat[2]|   |vtx.x|
// |vtx.y| = |mat[3] mat[4] mat[5]| * |vtx.y|
// |vtx.z|   |mat[6] mat[7] mat[8]|   |vtx.z|
Point& Point::Trans(const float* mat) {
	x = mat[0]*x + mat[1]*y + mat[2]*z;
	y = mat[3]*x + mat[4]*y + mat[5]*z;
	z = mat[6]*x + mat[7]*y + mat[8]*z;
	return *this;
}

void Point::Bound(float lowerBound, float upperBound) {
	if (x > upperBound) x = upperBound; if (x < lowerBound) x = lowerBound;
	if (y > upperBound) y = upperBound;	if (y < lowerBound) y = lowerBound;
	if (z > upperBound) z = upperBound;	if (z < lowerBound) z = lowerBound;
}

inline void Point::DrawVertex() {
	glVertex3f(x, y, z);
}

inline void Point::DrawNorm() {
	glNormal3f(x, y, z);
}