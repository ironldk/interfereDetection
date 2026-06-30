#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "Point.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat>

#define EPSILON 0.000000000001

#pragma pack(push, 1)
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
	
	__host__ __device__ bool Intersect(const Triangle& iTri) const;
	__host__ __device__ bool coplanar_tri_tri3d(const Point& iNormal, const Triangle& iTri) const;
	__host__ __device__ float ORIENT_2D(float a[2], float b[2], float c[2]) const;
	__host__ __device__ bool INTERSECTION_TEST_VERTEX(float P1[2], float Q1[2], float R1[2],
		float P2[2], float Q2[2], float R2[2]) const;
	__host__ __device__ bool INTERSECTION_TEST_EDGE(float P1[2], float Q1[2], float R1[2],
		float P2[2], float Q2[2], float R2[2]) const;
	__host__ __device__ bool ccw_tri_tri_intersection_2d(float p1[2], float q1[2], float r1[2],
		float p2[2], float q2[2], float r2[2]) const;
	__host__ __device__ bool tri_tri_overlap_test_2d(float p1[2], float p2[2]) const;

#ifndef DEBUG
private:
#endif
	Point a, b, c;
};
#pragma pack(pop)

// 静态断言验证Triangle的内存布局
static_assert(sizeof(Triangle) == 36, "Triangle size must be 36 bytes (3 Points x 12 bytes)");

inline __host__ __device__ Triangle& Triangle::Transform(float iMatrixModel[16]) {
	a.Transform(iMatrixModel);
	b.Transform(iMatrixModel);
	c.Transform(iMatrixModel);
	return *this;
}

__inline__ __host__ __device__ bool CHECK_MIN_MAX(const Point& p1, const Point& q1, const Point& r1,
	const Point& p2, const Point& q2, const Point& r2)
{	
	//      p1          p2
	//     / \         / \
	// ---i---j-------k---l------> L
	//   /     \     /     \
	//  r1-----q1   q2-----r2
	// to make [i,j] intersect [k,l], we need k<=j and i<=l
	return !(
	// j >= k means |p1,q1,p2,q2| <= 0, so
		(cross(q1 - p1, p2 - p1) * (q2 - p1) > 0.0f) +
	// i <= l means |p1,r1,r2,p2| <= 0, so
		(cross(r1 - p1, r2 - p1) * (p2 - p1) > 0.0f)
	);
}

// Permutation in a canonical form of T2's vertices
inline __host__ __device__ bool TRI_TRI_3D(const Point& p1, const Point& q1, const Point& r1,
	const Point& p2, const Point& q2, const Point& r2, float dp2, float dq2, float dr2)
{
	const Point *a1=&p1, *b1=&q1, *c1=&r1, *a2=&p2, *b2=&q2, *c2=&r2;
	if (dp2 > 0.0f) {
		if (dq2 > 0.0f) {
			b1=&r1; c1=&q1; a2=&r2; b2=&p2; c2=&q2;
		} else if (dr2 > 0.0f) {
			b1=&r1; c1=&q1; a2=&q2; b2=&r2; c2=&p2;
		}
	} else if (dp2 < 0.0f) {
		if (dq2 < 0.0f) {
			a2=&r2; b2=&p2; c2=&q2;
		} else if (dr2 < 0.0f) {
			a2=&q2; b2=&r2; c2=&p2;
		} else {
			b1=&r1; c1=&q1;
		}
	} else {
		if (dq2 < 0.0f) {
			if (dr2 >= 0.0f) {
				b1=&r1; c1=&q1; a2=&q2; b2=&r2; c2=&p2;
			}
		} else if (dq2 > 0.0f) {
			if (dr2 > 0.0f) {
				b1=&r1; c1=&q1;
			} else {
				a2=&q2; b2=&r2; c2=&p2;
			}
		} else {
			if (dr2 > 0.0f) {
				a2=&r2; b2=&p2; c2=&q2;
			} else if (dr2 < 0.0f) {
				b1=&r1; c1=&q1; a2=&r2; b2=&p2; c2=&q2;
			} else {
				// impossible case, coplanarity should have been handled before
			}
		}
	}
	return CHECK_MIN_MAX(*a1, *b1, *c1, *a2, *b2, *c2);
}

#define USE_EPSILON_TEST 1
// Guigue and Devillers triangle intersection algorithm
// https://inria.hal.science/inria-00072100/file/RR-4488.pdf
// https://github.com/erich666/jgt-code/blob/master/Volume_08/Number_1/Guigue2003/tri_tri_intersect.c
inline __host__ __device__ bool Triangle::Intersect(const Triangle& iT) const {
	const Point &p1=a, &q1=b, &r1=c, &p2=iT.a, &q2=iT.b, &r2=iT.c;
	// Compute distance signs of p1, q1 and r1 to the plane of triangle(p2, q2, r2)
	Point n2 = cross(q2 - p2, r2 - p2);
	float dp1 = n2 * (p1 - r2);
	float dq1 = n2 * (q1 - r2);
	float dr1 = n2 * (r1 - r2);

	// coplanarity robustness check
#if USE_EPSILON_TEST
	if (fabsf(dp1) < EPSILON) {dp1 = 0.0;}
	if (fabsf(dq1) < EPSILON) {dq1 = 0.0;}
	if (fabsf(dr1) < EPSILON) {dr1 = 0.0;}
#endif
	if (dp1 * dq1 > 0.0f && dp1 * dr1 > 0.0f) {
		return false;
	}

	// Compute distance signs of p2, q2 and r2 to the plane of triangle(p1, q1, r1)
	Point n1 = cross(q1 - p1, r1 - p1);
	float dp2 = n1 * (p2 - r1);
	float dq2 = n1 * (q2 - r1);
	float dr2 = n1 * (r2 - r1);
	// coplanarity robustness check
#if USE_EPSILON_TEST
	if (fabsf(dp2) < EPSILON) {dp2 = 0.0;}
	if (fabsf(dq2) < EPSILON) {dq2 = 0.0;}
	if (fabsf(dr2) < EPSILON) {dr2 = 0.0;}
#endif
	if (dp2 * dq2 > 0.0f && dp2 * dr2 > 0.0f) {
		return false;
	}

	const Point *a1=&p1, *b1=&q1, *c1=&r1, *a2=&p2, *b2=&q2, *c2=&r2;
	float da2=dp2, db2=dq2, dc2=dr2;
	if (dp1 > 0.0f) {
		if (dq1 > 0.0f) {
			a1=&r1; b1=&p1; c1=&q1; b2=&r2; c2=&q2; db2=dr2; dc2=dq2;
		} else if (dr1 > 0.0f) {
			a1=&q1; b1=&r1; c1=&p1; b2=&r2; c2=&q2; db2=dr2; dc2=dq2;
		}
	} else if (dp1 < 0.0f) {
		if (dq1 < 0.0f) {
			a1=&r1; b1=&p1; c1=&q1;
		} else if (dr1 < 0.0f) {
			a1=&q1; b1=&r1; c1=&p1;
		} else {
			b2=&r2; c2=&q2; db2=dr2; dc2=dq2;
		}
	} else {
		if (dq1 < 0.0f) {
			if (dr1 >= 0.0f) {
				a1=&q1; b1=&r1; c1=&p1; b2=&r2; c2=&q2; db2=dr2; dc2=dq2;
			}
		} else if (dq1 > 0.0f) {
			if (dr1 > 0.0f) {
				b2=&r2; c2=&q2; db2=dr2; dc2=dq2;
			} else {
				a1=&q1; b1=&r1; c1=&p1;
			}
		} else {
			if (dr1 > 0.0f) {
				a1=&r1; b1=&p1; c1=&q1;
			} else if (dr1 < 0.0f) {
				a1=&r1; b1=&p1; c1=&q1; b2=&r2; c2=&q2; db2=dr2; dc2=dq2;
			} else {
				return coplanar_tri_tri3d(n1, iT);
			}
		}
	}
	return TRI_TRI_3D(*a1, *b1, *c1, *a2, *b2, *c2, da2, db2, dc2);
}

// 共面三角形的2D重叠检测 - 基于Guigue和Devillers算法
inline __host__ __device__ bool Triangle::coplanar_tri_tri3d(const Point& iNormal, const Triangle& iTri) const {
	float arrX[2]{iNormal.x, -iNormal.x};
	float arrY[2]{iNormal.y, -iNormal.y};
	float arrZ[2]{iNormal.z, -iNormal.z};
	float n_x = arrX[iNormal.x < 0];
	float n_y = arrY[iNormal.y < 0];
	float n_z = arrZ[iNormal.z < 0];

	int maxAxis = 0; // 0 for x, 1 for y, 2 for z
	if (n_x > n_z) {
		if (n_x >= n_y) {
			maxAxis = 0;
		} else {
			maxAxis = 1;
		}
	} else { // n_z >= n_x
		if (n_z >= n_y) {
			maxAxis = 2;
		} else {
			maxAxis = 1;
		}
	}

	// Projection of the triangles in 3D onto 2D such that the area of the projection is maximized.
	float points2D[6][2];
	const Point* points[] = { &a, &b, &c, &iTri.a, &iTri.b, &iTri.c };
	for (int i = 0; i < sizeof(points)/sizeof(points[0]); ++i) {
		float coords[3]{ points[i]->x, points[i]->y, points[i]->z };
		for (int j = 0, k = 0; j < 2; ++j, ++k) {
			k = k + (j == maxAxis);
			coords[j] = coords[k];
		}
		points2D[i][0] = coords[0];
		points2D[i][1] = coords[1];
	}
	return tri_tri_overlap_test_2d(points2D[0], points2D[3]);
}

inline __host__ __device__ float Triangle::ORIENT_2D(float a[2], float b[2], float c[2]) const {
	return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
}

inline __host__ __device__ bool Triangle::INTERSECTION_TEST_VERTEX(float P1[2], float Q1[2], float R1[2],
	float P2[2], float Q2[2], float R2[2]) const
{
	if (ORIENT_2D(R2, P2, Q1) >= 0.0f) {
		if (ORIENT_2D(R2, Q2, Q1) <= 0.0f) {
			if (ORIENT_2D(P1, P2, Q1) > 0.0f) {
				return ORIENT_2D(P1, Q2, Q1) <= 0.0f;
			} else {
				if (ORIENT_2D(P1, P2, R1) >= 0.0f) {
					return ORIENT_2D(Q1, R1, P2) >= 0.0f;
				} else {
					return false;
				}
			}
		} else {
			if (ORIENT_2D(P1, Q2, Q1) <= 0.0f) {
				if (ORIENT_2D(R2, Q2, R1) <= 0.0f) {
					return ORIENT_2D(Q1, R1, Q2) >= 0.0f;
				} else {
					return false;
				}
			} else {
				return false;
			}
		}
	} else {
		if (ORIENT_2D(R2, P2, R1) >= 0.0f) {
			if (ORIENT_2D(Q1, R1, R2) >= 0.0f) {
				return ORIENT_2D(P1, P2, R1) >= 0.0f;
			} else {
				if (ORIENT_2D(Q1, R1, Q2) >= 0.0f) {
					return ORIENT_2D(R2, R1, Q2) >= 0.0f;
				} else {
					return false;
				}
			}
		} else {
			return false;
		}
	}
}

inline __host__ __device__ bool Triangle::INTERSECTION_TEST_EDGE(float P1[2], float Q1[2], float R1[2],
	float P2[2], float Q2[2], float R2[2]) const
{
	if (ORIENT_2D(R2, P2, Q1) >= 0.0f) {
		if (ORIENT_2D(P1, P2, Q1) >= 0.0f) {
			return ORIENT_2D(P1, Q1, R2) >= 0.0f;
		} else {
			if (ORIENT_2D(Q1, R1, P2) >= 0.0f) {
				return ORIENT_2D(R1, P1, P2) >= 0.0f;
			} else {
				return false;
			}
		}
	} else {
		if (ORIENT_2D(R2, P2, R1) >= 0.0f) {
			if (ORIENT_2D(P1, P2, R1) >= 0.0f) {
				if (ORIENT_2D(P1, R1, R2) >= 0.0f) {
					return true;
				} else {
					return ORIENT_2D(Q1, R1, R2) >= 0.0f;
				}
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
}

inline __host__ __device__ bool Triangle::ccw_tri_tri_intersection_2d(float p1[2], float q1[2], float r1[2],
	float p2[2], float q2[2], float r2[2]) const
{
	bool edge = false;
	bool p2q2left = ORIENT_2D(p2, q2, p1) >= 0.0f;
	bool q2r2left = ORIENT_2D(q2, r2, p1) >= 0.0f;
	bool r2p2left = ORIENT_2D(r2, p2, p1) >= 0.0f;
	if (p2q2left) {
		if (q2r2left) {
			if (r2p2left) {
				return true;
			} else {
				edge = true;
			}
		} else {
			if (r2p2left) {
				edge = true;
				float* temp = r2;
				r2 = q2; q2 = p2; p2 = temp;
			}
		}
	} else {
		if (q2r2left) {
			if (r2p2left) {
				edge = true;
			}
			float* temp = p2;
			p2 = q2; q2 = r2; r2 = temp;
		} else {
			float* temp = r2;
			r2 = q2; q2 = p2; p2 = temp;
		}
	}

	if (edge) {
		return INTERSECTION_TEST_EDGE(p1, q1, r1, p2, q2, r2);
	} else {
		return INTERSECTION_TEST_VERTEX(p1, q1, r1, p2, q2, r2);
	}
}

inline __host__ __device__ bool Triangle::tri_tri_overlap_test_2d(float p1[2], float p2[2]) const {
	int orient1 = (ORIENT_2D(p1, p1 + 2, p1 + 4) < 0.0f) * 2;
	int orient2 = (ORIENT_2D(p2, p2 + 2, p2 + 4) < 0.0f) * 2;
	return ccw_tri_tri_intersection_2d(p1, p1+2+orient1, p1+4-orient1, p2, p2+2+orient2, p2+4-orient2);
}

#endif