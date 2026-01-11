#ifndef OBJLOADER_H_
#define OBJLOADER_H_

#include <string>
#include <vector>
#include "Triangle.cuh"
#include "BBox.cuh"

class ObjLoader {
public:
	ObjLoader(const std::string& iStr);

	std::vector<Triangle> _tris;
};

#endif