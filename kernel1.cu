#pragma once
#include "freeglut.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include "CudaBVH.cuh"
#include "Vector3.h"

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
    Vector3 orig;
    Vector3 dir;
};
vector<Triangle> hitTris, BEG, END;

int *d_hit;
int *d_hit2;
int levelDisplay = 0;
//int yRot = 0;
float t = 0.287;
void display_cb() {
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(-0.5, -0.5, 0);
	//glRotated(yRot, 0.0, 1.0, 0.0); yRot++;
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
	glColor3f(1, 0, 0);
	for (auto &t : BEG) t.draw();
	glColor3f(0, 0, 1);
	for (auto &t : END) t.draw();
	glColor3f(0.8, 0.8, 0);
	for (auto &t : myBVH->myMesh) t.draw();
	glColor3f(0, 1, 0);
	for (int i = 0; i < myBVH2->myMesh.size(); ++i) {
		myBVH2->myMesh[i] = BEG[i] * (1 - t) + END[i] * t;
		myBVH2->myMesh[i].draw();
	}// t = 0.3;
	glEnd();
	// 重新生成包围盒
	myBVH2->myBBox.clear();
	for (auto f : myBVH2->myMesh) {
		Vector3 va(f.a.x, f.a.y, f.a.z);
		Vector3 vb(f.b.x, f.b.y, f.b.z);
		Vector3 vc(f.c.x, f.c.y, f.c.z);

		BBox b;
		b.xmin = fmin(fmin(va.x, vb.x), vc.x);
		b.xmax = fmax(fmax(va.x, vb.x), vc.x);
		b.ymin = fmin(fmin(va.y, vb.y), vc.y);
		b.ymax = fmax(fmax(va.y, vb.y), vc.y);
		b.zmin = fmin(fmin(va.z, vb.z), vc.z);
		b.zmax = fmax(fmax(va.z, vb.z), vc.z);
		myBVH2->myBBox.push_back(b);
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
    cout << float(myBVH->SAMPLE_SIZE) * 1e-6f / timer.interval() << " M detections / s" << endl;

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
        Vector3 v1 = sampleSphere(randf(), randf()) * 0.5f + Vector3(0.5, 0.5, 0.5);
        Vector3 v2 = sampleSphere(randf(), randf()) * 0.5f + Vector3(0.5, 0.5, 0.5);
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

    cout << hit << " out of " << myBVH->SAMPLE_SIZE + myBVH2->SAMPLE_SIZE << " triangles" << endl;

	//glPolygonMode(GL_FRONT, GL_FILL);
    for (int n = 0; n < hit; n++) {
        auto t = hitTris[n];
		glColor3f(1, 0, 0);
		glBegin(GL_TRIANGLES);
		glNormal3f(t.NormDir.x, t.NormDir.y, t.NormDir.z);
		glVertex3f(t.a.x, t.a.y, t.a.z);
		glVertex3f(t.b.x, t.b.y, t.b.z);
		glVertex3f(t.c.x, t.c.y, t.c.z);
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
//         Vector3 v1 = sampleSphere(nextFloat(), nextFloat());
//         Vector3 v2 = sampleSphere(nextFloat(), nextFloat());
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
//     cout << float(rays.size()) * 1e-6f / timer.interval() << " M rays / s" << endl;
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
	if (key == ']') t+=0.0001;
	if (key == '[') t-=0.0001;
    if (key == 'q') exit(0);
}

///
int sample_count = 0, sample_count2 = 0;
const int blockSize = 128;

// read file s, generate aabbs and triangles, then bound them both to (0,1)
void preprocess(string s, vector<BBox> &aabbs, vector<Triangle> &tris, int &sample_count, float3 trans) {
	obj *mesh = new obj;
	objLoader obj(s, mesh);

	auto faces = mesh->getFaces();
	auto verts = mesh->getPoints();

	vector<Vector3> my_verts;
	for (auto v : *verts) {
		my_verts.push_back(Vector3(v.x, v.y, v.z));
	}


	for (auto f : *faces) {
		if (f.size() == 3) {
			Vector3 va = my_verts[f[0]];
			Vector3 vb = my_verts[f[1]];
			Vector3 vc = my_verts[f[2]];

			BBox b;
			b.xmin = fmin(fmin(va.x, vb.x), vc.x);
			b.xmax = fmax(fmax(va.x, vb.x), vc.x);
			b.ymin = fmin(fmin(va.y, vb.y), vc.y);
			b.ymax = fmax(fmax(va.y, vb.y), vc.y);
			b.zmin = fmin(fmin(va.z, vb.z), vc.z);
			b.zmax = fmax(fmax(va.z, vb.z), vc.z);
			aabbs.push_back(b);

			Triangle t;
			t.a = make_float3(va.x, va.y, va.z);
			t.b = make_float3(vb.x, vb.y, vb.z);
			t.c = make_float3(vc.x, vc.y, vc.z);
			tris.push_back(t);
		}
	}
	sample_count = aabbs.size();

	// must be bounded to unit cube
	float bounds[6] = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	for (auto& b : aabbs) {
		bounds[0] = fmin(bounds[0], b.xmin);
		bounds[1] = fmax(bounds[1], b.xmax);
		bounds[2] = fmin(bounds[2], b.ymin);
		bounds[3] = fmax(bounds[3], b.ymax);
		bounds[4] = fmin(bounds[4], b.zmin);
		bounds[5] = fmax(bounds[5], b.zmax);
	}

	float _scale = fmin(fmin(1.0f / (bounds[1] - bounds[0]), 1.0f / (bounds[3] - bounds[2])), 1.0f / (bounds[5] - bounds[4]));
	for (auto& b : aabbs) {
		float lowerBound = 0, upperBound = 1;
		b.xmin = fmax(lowerBound, fmin(upperBound, (b.xmin - bounds[0]) * _scale));
		b.xmax = fmax(lowerBound, fmin(upperBound, (b.xmax - bounds[0]) * _scale));
		b.ymin = fmax(lowerBound, fmin(upperBound, (b.ymin - bounds[2]) * _scale));
		b.ymax = fmax(lowerBound, fmin(upperBound, (b.ymax - bounds[2]) * _scale));
		b.zmin = fmax(lowerBound, fmin(upperBound, (b.zmin - bounds[4]) * _scale));
		b.zmax = fmax(lowerBound, fmin(upperBound, (b.zmax - bounds[4]) * _scale));
		b += trans;
	}
	for (auto& t : tris) {
		((t -= make_float3(bounds[0], bounds[2], bounds[4])) *= _scale )+= trans;
	}
}
static void main_menu_func(int i) {}
int main(int argc, char **argv)
{
#if 1
	vector<BBox> aabbs, aabbs2;
	vector<Triangle> tris, tris2;
	preprocess("object1.stl", aabbs, tris, sample_count, make_float3(0, 0, 0));
	preprocess("object2.stl", aabbs2, tris2, sample_count2, make_float3(0.8890, 0, 0));//32);
	for (auto t : tris) {
		t += make_float3(-2, 0, 0);
		BEG.push_back(t);
		t += make_float3(1.5, -0.5, -0.5); t.rotate(make_float3(-0.70710678118654752440084436210485, 0, 0.70710678118654752440084436210485), 0.78539816339744830961566084581988); t += make_float3(2.5, 0.5, 0.5);
		END.push_back(t);
	}

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
        aabbs[i].xmax = max(buf[0], buf[1]);
        aabbs[i].xmin = min(buf[0], buf[1]);
        aabbs[i].ymax = max(buf[2], buf[3]);
        aabbs[i].ymin = min(buf[2], buf[3]);
        aabbs[i].zmax = max(buf[4], buf[5]);
        aabbs[i].zmin = min(buf[4], buf[5]);
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
    gluLookAt(0, 0.5, 4, 0, 0, 0, 0, 1, 0);
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