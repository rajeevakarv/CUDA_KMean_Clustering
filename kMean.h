/************************************************************************/
// The purpose of this program is to perform K-Means Clustering.
//
// Author: Jason Lowden
// Date: April 26, 2012
// Course: 0306-724 - High Performance Architectures
//
// File: common.h
/************************************************************************/

#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <cuda.h>
#include <cuda_runtime_api.h>

struct Vector2 {
	float x, y;

	inline __device__ __host__ float distSq(Vector2& p) {
		return (p.x - x)*(p.x - x) + (p.y - y)*(p.y - y);
	}
};

struct Datapoint {
	Vector2 p;
	int cluster;
	bool altered;
};

void KMeansCPU(Datapoint* data, long n, Vector2* clusters, int k);

bool KMeansGPU( Datapoint* data, long n, Vector2* clusters, int k );


#endif
