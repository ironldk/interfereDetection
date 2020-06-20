#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "Point.h"
#include<istream>
using std::istream;

class Triangle {
	friend bool samePlane(const Triangle& t1, const Triangle& t2);
public:
	Triangle(Point pa = Point(), Point pb = Point(),
			 Point pc = Point(), Point n  = Point());
	Triangle(istream& is);
	Triangle operator+(const Triangle& t);
	Triangle operator*(float scale);
	void Draw();
	Triangle& operator*=(float scale);
	Triangle& operator+=(const Point& orig);
	Triangle& operator-=(const Point& orig);
	operator BBox() const;
	Triangle& Trans(const float* mat);
	bool samePlane(const Triangle& tri);
	__device__ __host__ bool InterSect(const Triangle& r);
private:
	Point a, b, c, NormDir;
};

__device__ __host__ bool samePlane(const Triangle& rhs);

#endif