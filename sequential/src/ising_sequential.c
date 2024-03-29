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

#include "ising.h"

//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/


void ising(int *G, double *w, int k, int n){

  // array to write the new state for each iteration
  int *new_G = (int *)malloc(n * n * sizeof(int));
  if(new_G == NULL) exit(EXIT_FAILURE);

  // for every iteration
  for(int h = 0; h < k; h++){


    // for every point
    for(int ip = 0; ip < n; ip++){
      for(int jp = 0; jp < n; jp++){

        // variable for summation of neighbor weighted spins
        double weighted_sum = 0;

        // for every neighbor
        for(int in = -2; in <= 2; in++){
          for(int jn = -2; jn <= 2; jn++){

            // add weighted spins
            // use the mod operator to satisfy the periodic boundary conditions
            weighted_sum += w(in + 2 , jn + 2) * G((ip + in + n) % n , (jp + jn + n) % n);

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

    // Swap new_G and G pointers for next iteration
    int *temp;
    temp = G;
    G = new_G;
    new_G = temp;

  }

  /* If pointers have been swapped odd number of times
     copy result back to original array (new_G points to original G) */
  if(k % 2 == 1){
    for(int i = 0; i < n * n; i++){
      new_G[i] = G[i];
    }
    free(G);
  }else{
    free(new_G);
  }
}
