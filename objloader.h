#ifndef OBJLOADER_H_
#define OBJLOADER_H_

#include <string>
#include <vector>
#include "Triangle.h"
using std::vector;
using std::string;

vector<Triangle> extractTris(vector<Triangle>& tris) {
	vector<Triangle> temp;
	for (auto i = tris.begin(); i != tris.end(); ++i)
		for (auto j = tris.begin(); j != tris.end(); ++j)
			if (i != j && samePlane(*i, *j))
				temp.push_back(*i);
	return temp;
}

class ObjLoader {
public:
	ObjLoader(const string& s, vector<Triangle>& tris, vector<BBox> bboxes);
};

#endif