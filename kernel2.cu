#pragma once
#include "freeglut.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <set>
#include <random>
#include "CudaBVH.cuh"
#include "Point.h"
using std::set;

double randf() {
	return (double)(rand() / (double)RAND_MAX);
}

using namespace std;
CudaBVH* myBVH = nullptr, *myBVH2 = nullptr;

struct Timer{
    LARGE_INTEGER _begin, _end, _freq;
    Timer() {
        QueryPerformanceFrequency(&_freq);
    }
    void tick() {
        QueryPerformanceCounter(&_begin);
    }
    void tock() {
        QueryPerformanceCounter(&_end);
    }
    float interval() {
        return (_end.QuadPart - _begin.QuadPart) * 1e-6f;
    }
};
Timer timer;

struct Ray {
    Point orig;
    Point dir;
};
vector<Triangle> hitTris, BEG, END, plane;

int *d_hit;
int *d_hit2;
int levelDisplay = 0;
//int yRot = 0;
float t = 0.55; float X = 0, Y = 0, Z = 0;
void display_cb() {
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(-0.5, -0.5, 0);
	//glRotated(yRot, 0.0, 1.0, 0.0); yRot++;
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
	/*glColor3f(1, 0, 0);
	for (auto &t : plane) t.Draw();*/
	/*for (auto &t : BEG) t.Draw();
	glColor3f(0, 0, 1);
	for (auto &t : END) t.Draw();*/
	glColor3f(0.8, 0.8, 0.0);
	for (auto &t : myBVH->myMesh) t.Draw();
	glColor3f(0, 1, 0);
	for (int i = 0; i < myBVH2->myMesh.size(); ++i) {
		//myBVH2->myMesh[i] = BEG[i] * (1 - t) + END[i] * t;
		myBVH2->myMesh[i].Draw();
	}
	glEnd();
	// 重新生成包围盒
	myBVH2->myBBox.clear();
	for (auto f : myBVH2->myMesh) {
		myBVH2->myBBox.push_back(BBox(f));
	}

	//myBVH->draw(levelDisplay);
	//myBVH2->draw(levelDisplay);
	//myBVH->drawTrianglesDEBUG();
	//myBVH2->drawTrianglesDEBUG();
	//myBVH->drawTriangles();

#if 1
    ///
    // #pragma omp parallel for schedule(dynamic)

    timer.tick();
    checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(myBVH2->d_myMesh, &myBVH2->myMesh[0], myBVH2->SAMPLE_SIZE * sizeof(Triangle), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(myBVH2->d_myBBox, &myBVH2->myBBox[0], myBVH2->SAMPLE_SIZE * sizeof(BBox), cudaMemcpyHostToDevice));

	//myBVH->boxIntersect(myBVH2->SAMPLE_SIZE, myBVH2->d_myBBox, myBVH2->d_myMesh, d_hit, d_hit2);
    checkCudaErrors(cudaDeviceSynchronize());
    timer.tock();
    //cout << float(myBVH->SAMPLE_SIZE) * 1e-6f / timer.interval() << " M detections / filePath" << endl;

    int hit = 0;
    hitTris.clear();
    hitTris.resize(myBVH->SAMPLE_SIZE + myBVH2->SAMPLE_SIZE);
    for (int n = 0; n < myBVH->SAMPLE_SIZE; n++)
        if (d_hit[n])
            hitTris[hit++] = myBVH->myMesh[n];
	for (int n = 0; n < myBVH2->SAMPLE_SIZE; n++)
		if (d_hit2[n])
			hitTris[hit++] = myBVH2->myMesh[n];
#else
    ///
    int hit = 0;
    rays.clear();
    rays.resize(raycount);

    timer.tick();
    for (int n = 0; n < raycount; n++) {
        Ray ray;
        Point v1 = sampleSphere(randf(), randf()) * 0.5f + Point(0.5, 0.5, 0.5);
        Point v2 = sampleSphere(randf(), randf()) * 0.5f + Point(0.5, 0.5, 0.5);
        ray.orig = v1;
        ray.dir = Normalize(v2 - v1);

        float t, u, v;
        int idx;
        if (myBVH->intersect(make_float3(ray.orig.x, ray.orig.y, ray.orig.z), make_float3(ray.dir.x, ray.dir.y, ray.dir.z), t, u, v, idx)) 
        {
            Ray tr;
            tr.orig = ray.orig;
            tr.dir = ray.orig + ray.dir * t;
            rays[hit++] = tr;
        }
    }
    timer.tock();

    cout << float(raycount) * 1e-6f / timer.interval() << " M rays / s" << endl;
#endif

    //cout << hit << " out of " << myBVH->SAMPLE_SIZE + myBVH2->SAMPLE_SIZE << " triangles" << endl;

	//glPolygonMode(GL_FRONT, GL_FILL);
    for (int n = 0; n < hit; ++n) {
        auto t = hitTris[n];
		glColor3f(1, 0, 0);
		glBegin(GL_TRIANGLES);
		t.Draw();
		glEnd();
    }
    glPopMatrix();

//     bvh->DebugDraw();
// 
//     rays.clear();
//     rays.resize(10000);
//     timer.tick();
//     // #pragma omp parallel for schedule(dynamic)
//     for (int n = 0; n < rays.size(); n++) {
//         Ray ray;
//         Point v1 = sampleSphere(nextFloat(), nextFloat());
//         Point v2 = sampleSphere(nextFloat(), nextFloat());
//         ray.orig = v1;
//         ray.dir = Normalize(v2 - v1);
//         float t, u, v, w, sgn;
//         uint32_t idx;
//         if (bvh->TraceRay(ray.orig, ray.dir, t, u, v, w, sgn, idx)) {
//             Ray tr;
//             tr.orig = ray.orig;
//             tr.dir = ray.orig + ray.dir * t;
//             rays[n] = tr;
//         }
//     }
//     timer.tock();
//     cout << float(rays.size()) * 1e-6f / timer.interval() << " M rays / filePath" << endl;
// 
//     //     cout << rays.size() << endl;
//     glBegin(GL_LINES);
//     glColor3f(0, 0, 1);
//     for (auto v : rays) {
//         glVertex3fv(Raw(v.orig));
//         glVertex3fv(Raw(v.dir));
//     }
//     glEnd();

    glutSwapBuffers();
    glutPostRedisplay();
}

void keyboard_cb(unsigned char key, int x, int y) {
	if (key == '=') ++levelDisplay;
	if (key == '-') --levelDisplay;
	if (key == ']') t += 0.001;
	if (key == '[') t -= 0.001;
	if (key == '\'') Y += 0.1;
	if (key == ';') Y -= 0.1;
	if (key == '/') Z += 0.1;
	if (key == '.') Z -= 0.1; //cout << t << endl;
    if (key == 'q') exit(0);
}

int sample_count = 0, sample_count2 = 0;
const int blockSize = 128;

// read file filePath, generate aabbs and triangles,
// then bound them both to (0,1)
void preprocess(
	const string& str, vector<BBox>& aabbs, vector<Triangle>& tris,
	int& sample_count, float scale,	float3 axis, float theta)
{
	ObjLoader obj(str, tris, aabbs);
	sample_count = aabbs.size();

	// must be bounded to unit cube
	BBox bounds(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX);
	for (auto& b : aabbs)
        bounds.MakeEnvelope(b);

    float _scale = 1.0f / bounds.LargestEdge();
    Point minBound = bounds.GetMin();
	for (auto& b : aabbs) {
        b -= minBound;
        b *= _scale * scale;
        b.Bound(0, 1);
		b.Rotate(axis, theta);
	}
	double c = cos(theta), s = sin(theta);
	float mat[] = {
		(1-c)*axis.x*axis.x + c       , (1-c)*axis.y*axis.x - s*axis.z, (1-c)*axis.z*axis.x + s*axis.y,
		(1-c)*axis.x*axis.y + s*axis.z, (1-c)*axis.y*axis.y + c       , (1-c)*axis.z*axis.y - s*axis.x,
		(1-c)*axis.x*axis.z - s*axis.y, (1-c)*axis.y*axis.z + s*axis.x, (1-c)*axis.z*axis.z + c
	};
	for (auto& t : tris) {
        t -= minBound;
        t *= (_scale * scale);
        t.Trans(mat);
	}
}

static void main_menu_func(int i) {}

int main(int argc, char **argv) {
#if 1
	vector<BBox> aabbs, aabbs2;
	vector<Triangle> tris, tris2;
	set<Triangle> planes;
	preprocess("nut.stl", aabbs, tris, sample_count, 8, make_float3(0, 0, 0), 0);
	preprocess("wrench.stl", aabbs2, tris2, sample_count2, 88, make_float3(0, 0, -1), 1.308996938995747);
	//plane = extractTris(tris2);
	/*for (auto t : tris2) {
		t += make_float3(-33, -6.1, 0);
		BEG.push_back(t);
		t += make_float3(80, 0, 0);
		END.push_back(t);
	} */

     //for (auto b : aabbs) {
     //    cout << b.toString() << endl;
     //}
#else
    vector<BBox> aabbs(sample_count);
    float buf[6];
    for (int i = 0; i < sample_count; i++)
    {
        for (int j = 0; j < 6; j++)
            buf[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        aabbs[i]._max.x = max(buf[0], buf[1]);
        aabbs[i]._min.x = min(buf[0], buf[1]);
        aabbs[i]._max.y = max(buf[2], buf[3]);
        aabbs[i]._min.y = min(buf[2], buf[3]);
        aabbs[i]._max.z = max(buf[4], buf[5]);
        aabbs[i]._min.z = min(buf[4], buf[5]);
    }
#endif

    ///

	OutputDebugStringA(std::to_string(sample_count).c_str());
	myBVH = new CudaBVH(&aabbs[0], &tris[0], sample_count, blockSize);
	myBVH2 = new CudaBVH(&aabbs2[0], &tris2[0], sample_count2, blockSize);
//    system("pause");
//    BVHTree myTree = myBVH1->myTree;
//    for (int n = sample_count - 10; n < sample_count; n++) {
////         cout << "   aabb " << n << " : " << aabbs[myTree.leafNodes[n].getObjectID()].toString() << endl;
//
//        int idx = n;
//        MortonRec m = myBVH->mor[myTree.leafNodes[idx].getObjectID()];
//        printf("idx: %d  x: %f\n", idx, m.x);
//        printf("idx: %d  y: %f\n", idx, m.y);
//        printf("idx: %d  z: %f\n", idx, m.z);
//
//        printf("idx: %d  xx: %f\n", idx, m.xx);
//        printf("idx: %d  yy: %f\n", idx, m.yy);
//        printf("idx: %d  zz: %f\n", idx, m.zz);
//
//        printf("idx: %d  expand x: %lld\n", idx, m.ex);
//        printf("idx: %d  expand y: %lld\n", idx, m.ey);
//        printf("idx: %d  expand z: %lld\n", idx, m.ez);
//        printf("idx: %d  hash: %lld \n", idx, m.m);
//        printf("\n");
//    }
//        system("pause");
//		  myBVH->printBVH(myTree.internalNodes, myTree.leafNodes);
//        system("pause");

    checkCudaErrors(cudaMallocManaged((void**)&d_hit, myBVH->SAMPLE_SIZE * sizeof(int)));
    checkCudaErrors(cudaMallocManaged((void**)&d_hit2, myBVH2->SAMPLE_SIZE * sizeof(int)));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(2250, 1000);
    glutCreateWindow("Interfere Detection");
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(-8, 4, 64, -8, 4, 0, 0, 1, 0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, 2.25, 0.1, 100);
	glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display_cb);
    glutKeyboardFunc(keyboard_cb);
	glutCreateMenu(main_menu_func);
	
	glutAddMenuEntry("Interfere detection using old method", 0);
	glutAddMenuEntry("Interfere detection using our method", 1);
	glutAddMenuEntry("Update objectB's orientation", 2);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
    glutMainLoop();

	delete (myBVH); delete (myBVH2);
    checkCudaErrors(cudaFree(d_hit));
    checkCudaErrors(cudaFree(d_hit2));
    checkCudaErrors(cudaDeviceReset());
	return 0;
}