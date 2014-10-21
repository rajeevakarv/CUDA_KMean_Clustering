#include "KMeans.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>


#define TILE_SIZE 32   //start the block size.
#define MAX_BLOCKS_PER_GRID 65535   //Start the max grid size.

__constant__ Vector2 clustersconst[3];

/************************************************************************/
//
// GPU kernel for Kmean clustering.
//
/************************************************************************/
__global__ void KMeansKernelGPU( Datapoint* data, long n, int k )
{
	long blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; //calculate the threadId in grid. 
	if(threadId < n){			//check the condition. 
		data[threadId].altered = false;				// reset every element with altered 
		for(int clusterNo=0; clusterNo<k; clusterNo++)				//Loop over all the cluster
		{
			if(clustersconst[clusterNo].distSq(data[threadId].p) < clustersconst[data[threadId].cluster].distSq(data[threadId].p)){			//Check for distance with all the clusters.
				if(data[threadId].cluster != clusterNo){				//if elemets need to change the cluster.
					data[threadId].cluster = clusterNo;
					data[threadId].altered = true;						//change the altered value.
				}
			}
		}
	}
}


/************************************************************************/
// 
//	KmeanGPU method for kernel implementation of Kmean. 
//
/************************************************************************/
bool KMeansGPU( Datapoint* data, long n, Vector2* clusters, int k ){

	// Error return value
	cudaError_t status;						//Status variable for cuda call.
	long noElements =  n * sizeof(Datapoint);		//Total number of elements.
	Datapoint *dataDevice;							// Datapoints variable for device.
	status = cudaMalloc((void**) &dataDevice, noElements);		//Cuda melloc
	if (status != cudaSuccess)									//Error check
	{
		std::cout << "Kernel failed (Ad alloc): " << cudaGetErrorString(status) << 
					 std::endl;
		return false;
	}

	dim3 dimBlock(TILE_SIZE, 1);			//Initialize blocks.
	int gridx = 1;							//Vaiable for Grid size.
	int gridy = 1;							//Vaiable for Grid size
	//Logic to initialize 1D or 2D grids according to number of elements passed.
	if(n/TILE_SIZE < MAX_BLOCKS_PER_GRID+1)
		gridx = ceil((float)n/TILE_SIZE);
	else{
		gridx = MAX_BLOCKS_PER_GRID;
		gridy = ceil((float)n/(TILE_SIZE*MAX_BLOCKS_PER_GRID));
	}
	dim3 dimGrid(gridx, gridy);	 //Initialize grid size.
	bool is_KMean_done = true;			 //is_KMean_done to stop kmean clustring algo.
	
	while(is_KMean_done){
		is_KMean_done=false;			//initialize flag to false.
		status = cudaMemcpy(dataDevice, data, noElements, cudaMemcpyHostToDevice);			//Memcpy for data elements.
		if (status != cudaSuccess)					//Error check
		{
			std::cout << "Kernel failed (data Memcpy): " << cudaGetErrorString(status) << 
							std::endl;
			cudaFree(dataDevice);
			return false;
		}
	
		status = cudaMemcpyToSymbol(clustersconst, clusters, k*sizeof(Vector2), 0, cudaMemcpyHostToDevice);				//Copying cuda constant memory.
		if (status != cudaSuccess)																						//Error checking
		{
			std::cout << "Constant memory copy failed in const memory: " << cudaGetErrorString(status) << 
						 std::endl;
			cudaFree(dataDevice);
			return false;
		}

		KMeansKernelGPU<<<dimGrid, dimBlock>>>(dataDevice, n, k);				//Launching kernel
		// Wait for completion
		cudaThreadSynchronize();		

		// Check for errors
		status = cudaGetLastError();
		if (status != cudaSuccess)
		{
			std::cout << "Kernel failed on execution: " << cudaGetErrorString(status) << std::endl;
			cudaFree(dataDevice);
			return false;
		}
		
		status = cudaMemcpy(data, dataDevice, noElements, cudaMemcpyDeviceToHost);		//memcpy from device to host back the data points.
		if (status != cudaSuccess)														//Error check
		{
			std::cout << "Kernel failed (data Memcpy) cudaMemcpyDeviceToHost: " << cudaGetErrorString(status) << 
							std::endl;
			cudaFree(dataDevice);
			return false;
		}

		// Logic to update cluster.	
		for (int clusterNo=0; clusterNo<k; clusterNo++){						//loop over cluster.
			long numElements=0;													//initalize number of elements.
			for(long element=0; element<n; element++)							//Loop over elements.
			{
				if(data[element].cluster == clusterNo){							//check the data elements for cluster and update if needed.
					clusters[clusterNo].x += data[element].p.x;
					clusters[clusterNo].y += data[element].p.y;
					numElements++;
				}
				if(data[element].altered==true){								//Check for altered elements and update flag if nothing is updated.
					is_KMean_done=true;
				}
			}
			if(numElements>0){												//Update the mean to cluster point.
				clusters[clusterNo].x /= numElements;
				clusters[clusterNo].y /= numElements;
			}
		}
	}

	cudaFree(dataDevice);					//Free cuda mememory
	return true;
}
