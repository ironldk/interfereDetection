#include "objloader.h"
#include "Triangle.cuh"
#include "BBox.cuh"
#include <fstream>
using std::ifstream;

// read file filePath, generate aabbs and triangles
ObjLoader::ObjLoader(const std::string& iStr) {
	ifstream ifs(iStr);
	if (!ifs.is_open()) {
		return;
	}
	std::string str;
	ifs >> str;
	if (str != "solid") {
		return;
	}
	getline(ifs, str);

	// read Triangle
	for (int i = 0; ifs>>str && str=="facet"; ++i) {
		Triangle t;
		std::string str;

		getline(ifs, str);
		getline(ifs, str);
		
		ifs >> str;
		ifs >> t.a.x >> t.a.y >> t.a.z;
		ifs >> str;
		ifs >> t.b.x >> t.b.y >> t.b.z;
		ifs >> str;
		ifs >> t.c.x >> t.c.y >> t.c.z;

		ifs >> str;
		ifs >> str;

		_tris.push_back(std::move(t));
	}
}