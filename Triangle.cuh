#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "Point.cuh"
#define EPSILON 0.000000000001

class Triangle {
	friend class ObjLoader;
	friend class BBox;
public:
	Triangle();
	Triangle(Point pa, Point pb, Point pc);
	Triangle operator+(const Triangle& t);
	Triangle operator*(float scale);
	Triangle& operator*=(float scale);
	Triangle& operator+=(const Point& orig);
	Triangle& operator-=(const Point& orig);

	__host__ __device__ Triangle& Transform(float iMatrixModel[16]);

	__host__ __device__ float f_clamp(float n, float min, float max) const;
	
	__host__ __device__ float edge_to_edge(Point p1, Point q1,
		Point p2, Point q2, float& s, float& t, Point& c1, Point& c2) const;
	
	__host__ __device__ float point_to_triangle(Point pt, Point& ptt) const;
	
	// pr1,pr2 is nearest pair of points
	__host__ __device__ bool Intersect(const Triangle& r) const;

private:
	Point a, b, c;
};

inline __host__ __device__ Triangle& Triangle::Transform(float iMatrixModel[16]) {
	a.Transform(iMatrixModel);
	b.Transform(iMatrixModel);
	c.Transform(iMatrixModel);
	return *this;
}

inline __host__ __device__ float Triangle::f_clamp(float n, float min, float max) const {
	if (n < min) {
		return min;
	} else if (n > max) {
		return max;
	} else {
		return n;
	}
}

inline __host__ __device__ float Triangle::edge_to_edge(Point p1, Point q1,
	Point p2, Point q2, float& s, float& t, Point& c1, Point& c2) const
{
	Point d1 = q1 - p1; // Direction vector of segment S1
	Point d2 = q2 - p2; // Direction vector of segment S2
	Point r = p1 - p2;
	float a = d1 * d1; // Squared length of segment S1, always nonnegative
	float e = d2 * d2; // Squared length of segment S2, always nonnegative
	float f = d2 * r;
	// Check if either or both segments degenerate into points
	if (a <= EPSILON && e <= EPSILON) {
		// Both segments degenerate into points
		s = t = 0.0f;
		c1 = p1;
		c2 = p2;
		Point c2c1 = c1 - c2;
		return c2c1 * c2c1;
	} else if (a <= EPSILON) {
		// First segment degenerates into a point
		s = 0.0f;
		t = f / e; // s = 0 => t = (b*s + f) / e = f / e
		t = f_clamp(t, 0.0f, 1.0f);
	} else if (e <= EPSILON) {
		// Second segment degenerates into a point
		t = 0.0f;
		float c = d1 * r;
		s = f_clamp(-c / a, 0.0f, 1.0f); // t = 0 => s = (b*t - c) / a = -c / a
	} else {
		// The general nondegenerate case starts here
		float b = d1 * d2;
		float c = d1 * r;
		float denom = a * e - b * b; // Always nonnegative
		// If segments not parallel, compute closest point on L1 to L2 and
		// clamp to segment S1. Else pick arbitrary s (here 0)
		if (denom != 0.0f) {
			s = f_clamp((b * f - c * e) / denom, 0.0f, 1.0f);
		} else {
			s = 0.0f;
		}
		// Compute point on L2 closest to S1(s) using
		// t = Dot((P1 + D1*s) - P2,D2) / Dot(D2,D2) = (b*s + f) / e

		// If t in [0,1] done. Else clamp t, recompute s for the new value
		// of t using s = Dot((P2 + D2*t) - P1,D1) / Dot(D1,D1)= (t*b - c) / a
		// and clamp s to [0, 1]
		float tnom = b * s + f;
		if (tnom < 0.0f) {
			t = 0.0f;
			s = f_clamp(-c / a, 0.0f, 1.0f);
		}
		else if (tnom > e) {
			t = 1.0f;
			s = f_clamp((b - c) / a, 0.0f, 1.0f);
		}
		else {
			t = tnom / e;
		}
	}
	c1 = p1 + d1*s;
	c2 = p2 + d2*t;
	Point c2c1 = c1 - c2;
	return c2c1 * c2c1;
}

inline __host__ __device__ float Triangle::point_to_triangle(Point pt, Point& ptt) const {
	// Check if P in vertex region outside A
	Point ab = b - a;
	Point ac = c - a;
	Point ap = pt - a;
	float d1 = ab * ap;
	float d2 = ac * ap;
	if (d1 <= 0.0f && d2 <= 0.0f) {
		ptt = a;
		Point dist = pt - ptt;
		return dist * dist; // barycentric coordinates (1,0,0)
	}
	// Check if P in vertex region outside B
	Point bp = pt - b;
	float d3 = ab * bp;
	float d4 = ac * bp;
	if (d3 >= 0.0f && d4 <= d3) {
		ptt = b;
		Point dist = pt - ptt;
		return dist * dist; // barycentric coordinates (0,1,0)
	}
	// Check if P in edge region of AB, if so return projection of P onto AB
	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		ptt = a + ab*v;
		Point dist = pt - ptt;
		return dist * dist; // barycentric coordinates (1-v,v,0)
	}
	// Check if P in vertex region outside C
	Point cp = pt - c;
	float d5 = ab * cp;
	float d6 = ac * cp;
	if (d6 >= 0.0f && d5 <= d6) {// barycentric coordinates (0,0,1)
		ptt = c;
		Point dist = pt - ptt;
		return dist * dist;
	}
	// Check if P in edge region of AC, if so return projection of P onto AC
	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float w = d2 / (d2 - d6);
		ptt = a + ac*w;
		Point dist = pt - ptt;
		return dist * dist; // barycentric coordinates (1-w,0,w)
	}
	// Check if P in edge region of BC, if so return projection of P onto BC
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		ptt = b + (c-b)*w;
		Point dist = pt - ptt;
		return dist * dist; // barycentric coordinates (0,1-w,w)
	}
	// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	// = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
	ptt = a + ab*v + ac*w;
	Point dist = pt - ptt;
	return dist * dist;
}

inline __host__ __device__ bool Triangle::Intersect(const Triangle& r) const {
	Point pr1, pr2;
	float local_min;
	float temp;
	float s = 0, t = 0;
	Point p1, p2;
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
	temp = r.point_to_triangle(p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = b;
	temp = r.point_to_triangle(p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = c;
	temp = r.point_to_triangle(p1, p2);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = r.a;
	temp = point_to_triangle(p2, p1);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = r.b;
	temp = point_to_triangle(p2, p1);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = r.c;
	temp = point_to_triangle(p2, p1);
	if (temp < local_min) {
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	Point dist = pr1 - pr2;
	return (dist * dist) < EPSILON;
}

#endif