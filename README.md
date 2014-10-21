CUDA_KMean_Clustering
=====================

This repo is for CUDA implementation of KMean clustering.
A common problem in many fields is the classification of unknown data. Many automatic classification 
algorithms exist: hierarchical classifiers divide data up into a tree of categories, and clustering algorithms 
find optimal groupings of data into a fixed number of clusters. The K-means algorithm is a common and 
simple approach to clustering data in this way.
In this exercise, you will implement a K-means clustering algorithm that operates on two-dimensional 
data. This algorithm has two stages, one of which is parallelizable, so your program will need to deal 
with the transfer of data to and from the GPU during the course of the computation. Make sure you 
check for errors in memory allocation and kernel launch. The program should be capable of handling up 
to at least 220 (1M) samples; use long variables for indexing!
