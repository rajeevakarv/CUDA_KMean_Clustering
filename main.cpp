/************************************************************************/
// The purpose of this program is to perform K-MeansClustering.
//
// Author: Rajeev Verma
// Date: April 26, 2012
// Course: 0306-724 - High Performance Architectures
//
// File: main.cpp
/************************************************************************/

#include <cmath> // sqrt()
#include <ctime> // time(), clock()
#include <iostream> // cout, stream
#include <fstream>
#include "KMeans.h"

#define ITERS 1
#define DATA_SIZE (1<<25)

// To reset the cluster data between runs.
void initializeClusters(Vector2* clusters)
{
	clusters[0].x = 0;
	clusters[0].y = 0;

	clusters[1].x = 1;
	clusters[1].y = 0;

	clusters[2].x = -1;
	clusters[2].y = 0;
}

/************************************************************************/
//
// KMeansCPU method for K-mean clustering on CPU.
//
/************************************************************************/
void KMeansCPU(Datapoint* data, long n, Vector2* clusters, int k){
	long count;               //Count the number of elements change the cluster. 
	float mean=0;			//Variable used to temp store the mean
	do{
	count=0;                       //At every loop make count as zero.
	for(long i=0; i<n; i++)        //Loop over the elements
	{
		for(int clusterNo=0; clusterNo<k; clusterNo++)   //Loop over the clusters. 
		{
			if(clusters[clusterNo].distSq(data[i].p) < clusters[data[i].cluster].distSq(data[i].p)){      //Check if distance of current cluster is less. 
				data[i].cluster = clusterNo;																//Change value if condition is true
				count ++;																					//increament the count
			}
		}
	}
	//Now change the mean
	for (long clusterNo=0; clusterNo<k; clusterNo++){                //Loop for clusters.
			long numElements=0;
			for(long element=0; element<n; element++)                //Loop over eleemnts
			{
				if(data[element].cluster == clusterNo){                      //Check if cluster is same 
					clusters[clusterNo].x += data[element].p.x;				//sum-up the elements in x
					clusters[clusterNo].y += data[element].p.y;				//sum-up the elements in y
					numElements++;										   //take the count
				}
			}
		clusters[clusterNo].x /= numElements;								// change the element with mean in x
		clusters[clusterNo].y /= numElements;								// change the element with mean in y
		}
	}while(count!=0);    //End of do while
}

/************************************************************************/
// main method to call CPU and GPU functions and finally
//Present data to output.
/************************************************************************/
int main() 
{
	// The data we want to operate on.
	Datapoint* data = new Datapoint[DATA_SIZE];
	Datapoint* dataCPU = new Datapoint[DATA_SIZE];
    Datapoint* dataGPU = new Datapoint[ DATA_SIZE ];
	Vector2 clusters[3];

	std::cout << "Performing k-means clustering on " << DATA_SIZE << " values." << std::endl;
	
	// Fill up the example data using three gaussian distributed clusters.
	for (long i = 0; i < DATA_SIZE; i++) {
		int cluster = rand()%3;
		float u1 = (float)(rand()+1)/(float)RAND_MAX;
		float u2 = (float)(rand()+1)/(float)RAND_MAX;
		float z1 = sqrt(abs(-2 * log(u1))) * sin(6.283f*u2);
		float z2 = sqrt(abs(-2 * log(u1))) * cos(6.283f*u2);
		data[i].cluster = cluster; // ground truth
		switch (cluster) {						//fill up the data with random cluster
			case 0:
				data[i].p.x = z1;
				data[i].p.y = z2;
				break;
			case 1:
				data[i].p.x = 2 + z1 * 0.5f;
				data[i].p.y = 1 + z2 * 0.5f;
				break;
			case 2:
				data[i].p.x = -2 + z1 * 0.5f;
				data[i].p.y = 1 + z2 * 0.5f;
				break;
		}
	}


	float tcpu, tgpu;					//Timing variables.
	clock_t start, end;					//Clock start and end variables.
	long incorrect = 0;

	// Perform the host computations
	start = clock();					//Start of clock for CPU 
	for (int i = 0; i < ITERS; i++) {
		memcpy(dataCPU, data, sizeof(Datapoint) * DATA_SIZE);			//Copy the data points to CPU dataset
		initializeClusters(clusters);									//initialize cluster
		KMeansCPU(dataCPU, DATA_SIZE, clusters, 3);						//Call kmean for CPU for no of iterations.
	}
	end = clock();														//End time.
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;    //Calculate total time.

    
	for (long i = 0; i < DATA_SIZE; i++)
		if (data[i].cluster != dataCPU[i].cluster) incorrect++;	                           //calculate the data points change cluster.

	// Display the results for CPU
	std::cout << "Host Result took " << tcpu << " ms (" << (float)incorrect / (float)DATA_SIZE * 100 << "% misclassified)" << std::endl;

	for (int j = 0; j < 3; j++)
		std::cout << "Cluster " << j << ": " << clusters[j].x << ", " << clusters[j].y << std::endl;					// Prinint points for each cluster.
    std::cout << std::endl;

	//Processing for GPU.
	memcpy(dataGPU, data, sizeof(Datapoint) * DATA_SIZE);					//Memcopy for GPU implementation.
	initializeClusters(clusters);											//Initialize cluster.
	bool status = KMeansGPU( dataGPU, DATA_SIZE, clusters, 3);				//Kmean GPU method call.
	if(status==false)
	{
		std::cout <<"Cuda Kernel failed. to execute.\n"<<std::endl;			// handle error
		return 0;	
	}
   
	start = clock();						//Start of clock for GPU
	for (int i = 0; i < ITERS; i++) {
		memcpy(dataGPU, data, sizeof(Datapoint) * DATA_SIZE);			//Memcopy for GPU implementation.
		initializeClusters(clusters);									//Initialize cluster.
		KMeansGPU(dataGPU, DATA_SIZE, clusters, 3);						//kmean call for GPU.
	}
	end = clock();							//End of clock for GPU
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;			//timing calculation for GPU.
	incorrect = 0;														//reset incorrect
	for (long i = 0; i < DATA_SIZE; i++)
		if (data[i].cluster != dataGPU[i].cluster) incorrect++;			//check the elemets which change cluster.
	 
	// Display the results for GPU
	std::cout << "Device Result took " << tgpu << " ms (" << (float)incorrect / (float)DATA_SIZE * 100 << "% misclassified)" << std::endl;
	for (int j = 0; j < 3; j++)
		std::cout << "Cluster " << j << ": " << clusters[j].x << ", " << clusters[j].y << std::endl;
    std::cout << std::endl;

	//Write the results to a file.
    std::ofstream outfile("results.csv");
	outfile << "x,y,Truth,CPU,GPU" << std::endl;
	for (long i = 0; i < DATA_SIZE; i++) {
		outfile << data[i].p.x << "," << data[i].p.y << "," << data[i].cluster << "," << dataCPU[i].cluster << "," << dataGPU[i].cluster << "\n";
	}
	outfile.close();

	//delete the data
	delete[] data;
	delete[] dataCPU;
    delete[] dataGPU;

	getchar();
	// Success
	return 0;
}
