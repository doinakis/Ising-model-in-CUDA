/*
*Sequential Implementation
*Doinakis Michail && Paraskevas Thanos
*e-mail: doinakis@ece.auth.gr && athanasps@ece.auth.gr
       *
      ***
     *****
    *******
      |_|
*/

#include <stdio.h>
#include <stdlib.h>
#include "ising.h"


__global__
void ising_cuda(int *G, int *new_G, double *w, int n){

  int ip = blockIdx.x*blockDim.x + threadIdx.x;
  int jp = blockIdx.y*blockDim.y + threadIdx.y;


  if(ip < n && jp < n){

    // variable for summation of neighbor weighted spins
    double weighted_sum = 0.0f;

    // for every neighbor
    for(int in = -2; in <= 2; in++){
      for(int jn = -2; jn <= 2; jn++){

        // add weighted spins
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

void ising(int *G, double *w, int k, int n){

  int  *dev_new_G;

  // track initial pointer of G to write result back
  int *result = G;

  cudaMalloc(&dev_new_G, n*n*sizeof(int));

  // for every iteration
  for(int h = 0; h < k; h++){

    uint3 threadsPerBlock= make_uint3(1,1,1);
    uint3 blocksPerGrid = make_uint3(517,517,1);

    // call kernel function
    ising_cuda<<<blocksPerGrid, threadsPerBlock>>>(G, dev_new_G, w, n);

    // swap pointers
    int *temp = G;
    G = dev_new_G;
    dev_new_G = temp;

  }

  // copy result back to original array
  cudaMemcpy(result, dev_new_G, n*n, cudaMemcpyDeviceToDevice);

}
