#ifndef _POINT_H_
#define _POINT_H_

#include<istream>
class BBox;
using std::istream;

class Point {
	friend class Triangle;
	friend class BBox;
	friend Point operator*(float k, const Point& pt);
public:
	// constructors
	Point(float nx = 0.0, float ny = 0.0, float nz = 0.0);
	Point(const float3& pt): x(pt.x), y(pt.y), z(pt.z) {}
	Point(const Point& pt): x(pt.x), y(pt.y), z(pt.z) {}
	void read(istream &is);
	// overloaded operators
	Point operator-() const;
	Point operator*(float a) const;
	Point operator/(float a) const;
	Point& operator*=(float a);
	Point& operator/=(float a);
	Point operator+(const Point& pt) const;
	Point& operator+=(const Point& pt);
	Point& operator-=(const Point& pt);
	Point operator-(const Point& pt) const;
	float operator*(const Point& pt) const;

	operator float3();
	operator BBox() const;
	Point& Trans(const float* mat);
	void Bound(float lowerBound, float upperBound);
	void DrawVertex();
	void DrawNorm();
private:
	float x, y, z;
};
// friend
inline Point operator*(float k, const Point &pt) {
	return Point(k*pt.x, k*pt.y, k*pt.z);
}

#endif