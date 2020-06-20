#include "BBox.cuh"
#include <sstream>
#include "freeglut_std.h"
using std::stringstream;

__host__ __device__ BBox::BBox(float x0, float x1,
	float y0, float y1,	float z0, float z1)
:	_min(x0, y0, z0), _max(x1, y1, z1) {}

BBox::BBox(BBox a, BBox b, BBox c) {
	_min.x = min(min(a._min.x, b._min.x), c._min.x);
	_max.x = max(max(a._max.x, b._max.x), c._max.x);
	_min.y = min(min(a._min.y, b._min.y), c._min.y);
	_max.y = max(max(a._max.y, b._max.y), c._max.y);
	_min.z = min(min(a._min.z, b._min.z), c._min.z);
	_max.z = max(max(a._max.z, b._max.z), c._max.z);
}

void BBox::Draw(float* c) {
	glColor3f(c[0], c[1], c[2]);
	glPushMatrix();
	glBegin(GL_LINES);
	glVertex3f(_min.x, _max.y, _max.z);
	glVertex3f(_max.x, _max.y, _max.z);
	glVertex3f(_min.x, _min.y, _min.z);
	glVertex3f(_max.x, _min.y, _min.z);
	glVertex3f(_min.x, _min.y, _max.z);
	glVertex3f(_max.x, _min.y, _max.z);
	glVertex3f(_min.x, _max.y, _min.z);
	glVertex3f(_max.x, _max.y, _min.z);

	glVertex3f(_min.x, _min.y, _min.z);
	glVertex3f(_min.x, _max.y, _min.z);
	glVertex3f(_min.x, _min.y, _max.z);
	glVertex3f(_min.x, _max.y, _max.z);
	glVertex3f(_max.x, _min.y, _min.z);
	glVertex3f(_max.x, _max.y, _min.z);
	glVertex3f(_max.x, _min.y, _max.z);
	glVertex3f(_max.x, _max.y, _max.z);

	glVertex3f(_min.x, _min.y, _min.z);
	glVertex3f(_min.x, _min.y, _max.z);
	glVertex3f(_min.x, _max.y, _min.z);
	glVertex3f(_min.x, _max.y, _max.z);
	glVertex3f(_max.x, _min.y, _min.z);
	glVertex3f(_max.x, _min.y, _max.z);
	glVertex3f(_max.x, _max.y, _min.z);
	glVertex3f(_max.x, _max.y, _max.z);
	glEnd();
	glPopMatrix();
}

bool BBox::Contains(float3 point) {
	if (point.x <= _max.x && point.x >= _min.x &&
		point.y <= _max.y && point.y >= _min.y &&
		point.z <= _max.z && point.z >= _min.z)
		return true;
	else
		return false;
}

string BBox::toString() {
	stringstream ss;
	ss << "x range: (" << _min.x << "," << _max.x << ")\t";
	ss << "y range: (" << _min.y << "," << _max.y << ")\t";
	ss << "z range: (" << _min.z << "," << _max.z << ")\n";
	return ss.str();
}

BBox& BBox::operator+=(const Point& pt) {
	_min += pt;
	_max += pt;
	return *this;
}

BBox& BBox::operator-=(const Point& pt) {
	_min -= pt;
	_max -= pt;
	return *this;
}

BBox& BBox::operator*=(const float& scale) {
	_min *= scale;
	_max *= scale;
	return *this;
}

BBox& BBox::Rotate(float3 r, float t) {
	double sint = sin(t), cost = cos(t);
	float matRot[9] = {
(1-cost)*r.x*r.x+cost    ,(1-cost)*r.x*r.y+sint*r.z,(1-cost)*r.x*r.z-sint*r.y,
(1-cost)*r.y*r.x-sint*r.z,(1-cost)*r.y*r.y+cost    ,(1-cost)*r.y*r.z+sint*r.x,
(1-cost)*r.z*r.x+sint*r.y,(1-cost)*r.z*r.y-sint*r.x,(1-cost)*r.z*r.z+cost
	};
	_min.Trans(matRot);
	_max.Trans(matRot);
	return *this;
}

void BBox::MakeEnvelope(const BBox& b) {
	if (_min.x > b._min.x) _min.x = b._min.x;
	if (_max.x < b._max.x) _max.x = b._max.x;
	if (_min.y > b._min.y) _min.y = b._min.y;
	if (_max.y < b._max.y) _max.y = b._max.y;
	if (_min.z > b._min.z) _min.z = b._min.z;
	if (_max.z < b._max.z) _max.z = b._max.z;
}

bool BBox::Intersect(const BBox& b) const {
	// Exit with no intersection if separated along an axis
	if (_max.x < b._min.x || _min.x > b._max.x) return false;
	if (_max.y < b._min.y || _min.y > b._max.y) return false;
	if (_max.z < b._min.z || _min.z > b._max.z) return false;
	// Overlapping on all axes means AABBs are intersecting
	return true;
}

float BBox::LargestEdge() {
	Point pt = _max - _min;
	float largest = pt.x;
	if (largest < pt.y)
		largest = pt.y;
	if (largest < pt.z)
		largest = pt.z;
	return largest;
}

const Point& BBox::GetMin() {
	return _min;
}

void BBox::Bound(float lowerBound, float upperBound) {
	_min.Bound(lowerBound, upperBound);
	_max.Bound(lowerBound, upperBound);
}

void BBox::GetCenter(float& ox, float& oy, float& oz) const {
	ox = (_min.x + _max.x) / 2;
	oy = (_min.y + _max.y) / 2;
	oz = (_min.z + _max.z) / 2;
}
