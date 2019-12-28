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

  int k=1, n = 517;
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
      w(i,j) = a[i][j];
    }
  }

  // Allocate spins array
  int *G = (int *)malloc(n * n * sizeof(int));
  if(G == NULL) exit(EXIT_FAILURE);

  size_t size;

  //load initial state of spins
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

  // ========== TESTER ==========
  // ========== k = 1 ==========
  k = 1;
  flag = 0;

  ising(G, w, k, n);

  for(int i=0;i<20;i++){
    for(int j=0;j<10;j++){
      printf("%2d\t", G(i,j));
    }
    printf("\n");
  }

  int *test = (int *)malloc(n * n * sizeof(int));
  fp = fopen("inc/conf-1.bin", "rb");
  size = fread(test, sizeof(int), n * n, fp);
  if(size!=n*n) exit(EXIT_FAILURE);
  fclose(fp);

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      if (G(i,j) != test(i,j)) {
        printf("k = %d - WRONG\n", k);
        flag = 1;
        break;
      }
    }
    if(flag)
      break;
  }

  if(!flag)
    printf("k = %d - CORRECT\n", k);


  return 0;

}
