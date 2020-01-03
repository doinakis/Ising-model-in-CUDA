/*
*Cuda Implementation #1
*Doinakis Michail && Paraskevas Thanos
*e-mail: doinakis@ece.auth.gr && athanasps@ece.auth.gr
       *
      ***
     *****
    *******
      |_|
*/

#include "ising.h"


//! Ising model evolution KERNEL
/*!

  \param G        Spins on the square lattice                     [n-by-n]
  \param new_G    Spins on the square lattice at next time step   [n-by-n]
  \param w        Weight matrix                                   [5-by-5]
  \param n        Number of lattice points per dim                [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/

__global__
void ising_kernel(int *G, int *new_G, double *w, int n){

  // assing a point to every thread (1 thread per block) to match the Ising model lattice
  int ip = blockIdx.x*blockDim.x + threadIdx.x;
  int jp = blockIdx.y*blockDim.y + threadIdx.y;


  if(ip < n && jp < n){

    // variable for summation of neighbor weighted spins
    double weighted_sum = 0.0f;

    // for every neighbor
    for(int in = -2; in <= 2; in++){
      for(int jn = -2; jn <= 2; jn++){

        // add weighted spins
        // use the mod operator to satisfy the periodic boundary conditions
        weighted_sum += w(in + 2 , jn + 2) * G((ip + in + n) % n, (jp + jn + n) % n);

      }
    }

    // precision to account for floating point errors
    double epsilon = 1e-4;

    // Update magnetic momment
    if(weighted_sum > epsilon){

      new_G(ip,jp) = 1;

    }else if(weighted_sum < - epsilon){

      new_G(ip,jp) = -1;

    }else{

      new_G(ip,jp) = G(ip,jp);

    }

  }
}


//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/

// === This is a WRAPPER for calling the cuda kernel ===

void ising(int *G, double *w, int k, int n){

  // DEVICE ALLOCATION
  int *dev_G, *dev_new_G;
  double *dev_w;

  cudaMalloc(&dev_G, n*n*sizeof(int));
  cudaMalloc(&dev_new_G, n*n*sizeof(int));
  cudaMalloc(&dev_w, 5*5*sizeof(double));

  // tranfer data to device
  cudaMemcpy(dev_G, G, n*n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_w, w, 5*5*sizeof(double), cudaMemcpyHostToDevice);


  // grid and blocks dimensions
  uint3 threadsPerBlock= make_uint3(1,1,1);
  uint3 blocksPerGrid = make_uint3(n,n,1);

  // for every iteration
  for(int h = 0; h < k; h++){

    // call kernel function
    ising_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_G, dev_new_G, dev_w, n);

    // swap pointers
    int *temp = dev_G;
    dev_G = dev_new_G;
    dev_new_G = temp;

  }

  // copy result back to original array
  cudaMemcpy(G, dev_G, n*n*sizeof(int), cudaMemcpyDeviceToHost);

  // free device arrays
  cudaFree(dev_G);
  cudaFree(dev_new_G);
  cudaFree(dev_w);

}
