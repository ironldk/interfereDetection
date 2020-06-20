#include "Triangle.h"

Triangle::Triangle(Point pa, Point pb, Point pc, Point n):
	a(pa), b(pb), c(pc), NormDir(n) {}

Triangle::Triangle(istream &is) {
	string str;
	is >> str;//get rid of "normal "
	NormDir.read(is);
	NormDir = -NormDir;
	//get rid of "outer loop"
	getline(is, str);
	getline(is, str);
	is >> str;
	a.read(is);
	is >> str;
	b.read(is);
	is >> str;
	c.read(is);
}
Triangle Triangle::operator+(const Triangle& t) {
	return Triangle(a+t.a, b+t.b, c+t.c);
}
Triangle Triangle::operator*(float scale) {
	return Triangle(a*scale, b*scale, c*scale);
}
inline void Triangle::Draw() {
	NormDir.DrawNorm();
	a.DrawVertex();
	b.DrawVertex();
	c.DrawVertex();
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

Triangle::operator BBox() const {
	return BBox(BBox(a), BBox(b), BBox(c));
}

Triangle& Triangle::Trans(const float* mat) {
	a.Trans(mat);
	b.Trans(mat);
	c.Trans(mat);
	return *this;
}

__device__ __host__ bool samePlane(const Triangle& rhs) {
	if ((rhs.NormDir - NormDir) * (rhs.NormDir - NormDir) < EPSILON &&
		abs(a * NormDir - rhs.a * NormDir) < EPSILON) {
		return true;
	} else {
		return false;
	}
}

__device__ __host__ bool Triangle::InterSect(const Triangle& r) {
	float3 pr1, pr2;
	float local_min;
	float temp;
	float s = 0, t = 0;
	float3 p1;
	float3 p2;
	local_min = edge_to_edge(a, b, r.a, r.b, s, t, p1, p2);
	pr1 = p1;
	pr2 = p2;
	temp = edge_to_edge(a, c, r.a, r.b, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c, b, r.a, r.b, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a, b, r.a, r.c, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a, c, r.a, r.c, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c, b, r.a, r.c, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a, b, r.c, r.b, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a, c, r.c, r.b, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c, b, r.c, r.b, s, t, p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}

	p1 = a;
	temp = point_to_triangle(p1, p2, r.a, r.b, r.c);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = b;
	temp = point_to_triangle(p1, p2, r.a, r.b, r.c);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = c;
	temp = point_to_triangle(p1, p2, r.a, r.b, r.c);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = r.a;
	temp = point_to_triangle(p2, p1, a, b, c);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = r.b;
	temp = point_to_triangle(p2, p1, a, b, c);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = r.c;
	temp = point_to_triangle(p2, p1, a, b, c);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	float3 dist = f3_sub(pr1, pr2);
	return (f3_dot(dist, dist) < EPSILON);
}