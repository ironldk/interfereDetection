#ifndef BBOX_CUH_
#define BBOX_CUH_

#include "Point.h"
#include <string>
using std::string;

class BBox {
public:
	__host__ __device__ BBox(
		float x0 = 0.0, float x1 = 0.0,
		float y0 = 0.0, float y1 = 0.0,
		float z0 = 0.0, float z1 = 0.0
	);
	BBox(BBox a, BBox b, BBox c);
	void Draw(float* c);
	bool Contains(float3 point);
	string toString();
	BBox& operator+=(const Point& pt);
	BBox& operator-=(const Point& pt);
	BBox& operator*=(const float& scale);
	BBox& Rotate(float3 r, float t);
	void MakeEnvelope(const BBox& bbox);
	bool Intersect(const BBox& b) const;
	float LargestEdge();
	const Point& GetMin();
	void Bound(float lowerBound, float upperBound);
	void GetCenter(float& ox, float& oy, float& oz) const;
private:
	Point _min, _max;
};

#endif