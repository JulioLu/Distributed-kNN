# Distributed knn

recursive

k-NN - Distributed-MPI

Function V0

Function V0 implements the basic idea of ​​finding nearest neighbors (k-NN). Uses high performance BLAS routines. Once a k-NN is found, the "work ()" method is used, which renews the k-NNs while maintaining their classification.

Function V1

The V1 function distributes the calculation process of V0's nearest neighbors to many machines. Using the following method: The first procedure distributes all the data to the other "processes" then each process locally calculates the nearest neighbors of its points using V0. Specifically in each step each process holding firmly its initial points (which it accepted from the first procedure) as query points accepts the points of the previous one and sends its own to the next procedure.

Using V0, the nearest neighbors are calculated using the data it receives as corpus points and its initial points as query points. At each step it renews the nearest neighbors of its fixed query points.

Function V2

The V2 function distributes the nearest neighbor calculation (k-NN) process based on the creation of a Vantage Point Tree. Where each node of the tree is a point of the original set. This way the search for a neighbor is done in logN steps and the total search of k-NN in NlogN steps. The search () function in each retro call renews the nearest neighbors of the point using the work () method. The V2 function then uses the encoder () method which encodes the subtree to be sent to the next prosses and the decoder () method which decodes the data received by the current prosses from the previous one. Finally the V2 function fills the knnresult structure from the struct pointData table.


