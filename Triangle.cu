#include "Triangle.cuh"
#include "BBox.cuh"

Triangle::Triangle() {}

Triangle::Triangle(Point pa, Point pb, Point pc):
	a(pa), b(pb), c(pc) {}

Triangle Triangle::operator+(const Triangle& t) {
	return Triangle(a+t.a, b+t.b, c+t.c);
}
Triangle Triangle::operator*(float scale) {
	return Triangle(a*scale, b*scale, c*scale);
}

Triangle& Triangle::operator*=(float scale) {
	a *= scale;
	b *= scale;
	c *= scale;
	return *this;
}
Triangle& Triangle::operator+=(const Point& orig) {
	a += orig;
	b += orig;
	c += orig;
	return *this;
}
Triangle& Triangle::operator-=(const Point& orig) {
	a -= orig;
	b -= orig;
	c -= orig;
	return *this;
}