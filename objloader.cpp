#include "objloader.h"
#include <fstream>
using std::ifstream;

ObjLoader::ObjLoader(const string& s, vector<Triangle>& tris, vector<BBox> bboxes) {
	ifstream ifs(s);
	if (!ifs.is_open()) {
		return;
	}
	string str;
	ifs >> str;
	if (str != "solid") {
		return;
	}
	getline(ifs, str);
	for (int i = 0; ifs >> str && str == "facet"; ++i) {
		Triangle t(ifs);
		tris.push_back(t);
		bboxes.push_back((BBox)t);
		ifs >> str;
		ifs >> str;
		ifs >> str;
	}
}