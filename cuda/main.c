#include <stdio.h>
#include <stdlib.h>
#include "ising.h"

int main(int argc, char *argv[]){

  int k = 1;
  int n = 517;

  double *w = (double *)malloc(5 * 5 * sizeof(double));
  if(w == NULL) exit(EXIT_FAILURE);

  int *G = (int *)malloc(n * n * sizeof(int));
  if(G == NULL) exit(EXIT_FAILURE);

  size_t size;


  FILE *fp = fopen("inc/conf-init.bin", "rb");
  size = fread(G, sizeof(int), n * n, fp);
  if(size!=n*n) exit(EXIT_FAILURE);
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
  fp = fopen("inc/conf-1.bin", "rb");
  size = fread(test, sizeof(int), n * n, fp);
  if(size!=n*n) exit(EXIT_FAILURE);
  fclose(fp);

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      if (G(i,j) != test(i,j)) {
        printf("WRONG at (%d,%d)\n", i, j);
        exit(EXIT_FAILURE);
      }
    }
  }

  printf("CORRECT\n");

  return 0;

}
