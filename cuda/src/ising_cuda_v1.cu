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

//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/

__global__
void ising(int *G, int *new_G, double *w, int k, int n){

  int ip = blockIdx.x*blockDim.x + threadIdx.x;
  int jp = blockIdx.y*blockDim.y + threadIdx.y;

  // track initial pointer of G to write result back
  int *result = G;

  if(ip < n && jp < n){

    // for every iteration
    for(int h = 0; h < k; h++){

      // variable for summation of neighbor weighted spins
      double weighted_sum = 0.0;

      // for every neighbor
      for(int in = -2; in <= 2; in++){
        for(int jn = -2; jn <= 2; jn++){

          // add weighted spins
          int a = (ip + in + n) % n;
          int b = (jp + jn + n) % n;
          weighted_sum +=  G(a, b);//w(in + 2 , jn + 2) *
          __syncthreads();

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
      new_G(ip,jp) = weighted_sum;


      // Swap new_G and G pointers for next iteration
      int *temp;
      temp = G;
      G = new_G;
      new_G = temp;

    }
    // copy result back to original array
    result(ip,jp) = G(ip,jp);
  }
}
