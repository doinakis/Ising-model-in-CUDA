#include <stdio.h>
#include <stdlib.h>
#include "ising.h"

int main(){
  int k = 1;
  int n = 517;
  double *w = (double *)malloc(5 * 5 * sizeof(double));
  int *G = (int *)malloc(n * n * sizeof(int));
  size_t size;


  FILE *fp = fopen("conf-init.bin", "rb");
  size = fread(G, sizeof(int), n * n, fp);
  if(size!=n*n) exit(1);
  fclose(fp);
  // for(int i = 0; i < 4; i++){
  //   for(int j = 0; j < 4; j++){
  //     printf("%d  ",G(i,j));
  //   }
  // }

  double a[5][5] = {
   {0.004, 0.016, 0.026, 0.016, 0.004} ,
   {0.016, 0.071, 0.117, 0.071, 0.016} ,
   {0.026, 0.117, 0.000,  0.117, 0.026} ,
   {0.016, 0.071, 0.117, 0.071, 0.016} ,
   {0.004, 0.016, 0.026, 0.016, 0.004}
  };
  for(int i = 0; i < 5; i++){
    for(int j = 0; j < 5; j++){
      w(i,j) = a[i][j];
    }
  }

  ising(G, w, k, n);

  int *test = (int *)malloc(n * n * sizeof(int));
  fp = fopen("conf-1.bin", "rb");
  size = fread(test, sizeof(int), n * n, fp);
  if(size!=n*n) exit(1);
  fclose(fp);

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      if (G(i,j) != test(i,j)) {
        printf("WRONG at (%d,%d)\n", i, j);
        // exit(1);
      }
    }
  }

  printf("CORRECT\n");

  return 0;

}
