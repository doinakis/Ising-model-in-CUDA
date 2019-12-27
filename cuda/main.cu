/*
*Sequential Main function with tester
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

int main(int argc, char *argv[]){

  int k, n = 517;
  int flag;

  // HOST ALLOCATION
  // Allocate weights array
  double *w = (double *)malloc(5 * 5 * sizeof(double));
  if(w == NULL) exit(EXIT_FAILURE);

  double a[5][5] = {
   {0.004, 0.016, 0.026, 0.016, 0.004} ,
   {0.016, 0.071, 0.117, 0.071, 0.016} ,
   {0.026, 0.117, 0.000, 0.117, 0.026} ,
   {0.016, 0.071, 0.117, 0.071, 0.016} ,
   {0.004, 0.016, 0.026, 0.016, 0.004}
  };
  for(int i = 0; i < 5; i++){
    for(int j = 0; j < 5; j++){
      w(i,j) = 1;//a[i][j];
    }
  }

  // Allocate spins array
  int *G = (int *)malloc(n * n * sizeof(int));
  if(G == NULL) exit(EXIT_FAILURE);

  size_t size;

  // load initial state of spins
  FILE *fp = fopen("inc/conf-init.bin", "rb");
  size = fread(G, sizeof(int), n * n, fp);
  if(size!=n*n) exit(EXIT_FAILURE);
  fclose(fp);

  for(int i=0;i<10;i++){
    for(int j=0;j<10;j++){
      printf("%2d\t", G(i,j));
    }
    printf("\n");
  }

  printf("\n\n");

  // DEVICE ALLOCATION
  int *dev_G, *dev_new_G;
  double *dev_w;


  cudaMalloc(&dev_G, n*n*sizeof(int));
  cudaMalloc(&dev_w, 5*5*sizeof(double));
  cudaMalloc(&dev_new_G, n*n*sizeof(int));

  // tranfer data to device
  cudaMemcpy(dev_G, G, n*n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_w, w, 5*5*sizeof(double), cudaMemcpyHostToDevice);


  // ========== TESTER ==========
  // ========== k = 1 ==========
  k = 1;
  flag = 0;
//  uint3 threadsPerBlock= make_uint3();
  uint3 blocksPerGrid = make_uint3(517,517,1);

  //calling kernel function
  ising<<<blocksPerGrid,1>>>(dev_G, dev_new_G, dev_w, k, n);
  cudaDeviceSynchronize();

  cudaMemcpy(G, dev_G, n*n*sizeof(int), cudaMemcpyDeviceToHost);

  for(int i=0;i<20;i++){
    for(int j=0;j<10;j++){
      printf("%2d\t", G(i,j));
    }
    printf("\n");
  }


  return 0;

}
