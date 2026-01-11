#include "BBox.cuh"

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

float BBox::LargestEdge() {
	float dx = _max.x - _min.x,
		  dy = _max.y - _min.y,
		  dz = _max.z - _min.z;
	if (dx < dy) {
		if (dz < dy) {
			return dy;
		} else {
			return dz;
		}
	} else {
		if (dz < dx) {
			return dx;
		} else {
			return dz;
		}
	}
}

const Point& BBox::GetMin() {
	return _min;
}