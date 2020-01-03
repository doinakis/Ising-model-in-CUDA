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

#include "ising.h"

int main(int argc, char *argv[]){

  int k, n = 517;
  int flag;

  // allocate weights array
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

  // allocate spins array
  int *G = (int *)malloc(n * n * sizeof(int));
  if(G == NULL) exit(EXIT_FAILURE);

  size_t size;

  // load initial state of spins
  FILE *fp = fopen("inc/conf-init.bin", "rb");
  size = fread(G, sizeof(int), n * n, fp);
  if(size!=n*n) exit(EXIT_FAILURE);
  fclose(fp);

  // ========== TESTER ==========
  // ========== k = 1 ==========
  k = 1;
  // helper variable to stop the iterations if a wrong result is found
  flag = 0;

  ising(G, w, k, n);

  // load expected state of spins after k iterations
  int *test = (int *)malloc(n * n * sizeof(int));
  fp = fopen("inc/conf-1.bin", "rb");
  size = fread(test, sizeof(int), n * n, fp);
  if(size!=n*n) exit(EXIT_FAILURE);
  fclose(fp);

  // for every point
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      // if the result doesn't match the expected result print WRONG
      if (G(i,j) != test(i,j)) {
        printf("k = %d - WRONG\n", k);
        flag = 1;
        break;
      }
    }
    if(flag)
      break;
  }
  //
  if(!flag)
    printf("k = %d - CORRECT\n", k);


  // ========== k = 4 ==========
  k = 4;
  flag = 0;

  // reload inital state of spins
  fp = fopen("inc/conf-init.bin", "rb");
  size = fread(G, sizeof(int), n * n, fp);
  if(size!=n*n) exit(EXIT_FAILURE);
  fclose(fp);

  ising(G, w, k, n);

  // load expected state of spins after k iterations
  fp = fopen("inc/conf-4.bin", "rb");
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

  // ========== k = 11 ==========
  k = 11;
  flag = 0;

  // reload inital state of spins
  fp = fopen("inc/conf-init.bin", "rb");
  size = fread(G, sizeof(int), n * n, fp);
  if(size!=n*n) exit(EXIT_FAILURE);
  fclose(fp);

  ising(G, w, k, n);

  // load expected state of spins after k iterations
  fp = fopen("inc/conf-11.bin", "rb");
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

  free(G);
  free(w);
  free(test);

  return 0;

}
