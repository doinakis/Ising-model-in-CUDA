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
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "ising.h"


int main(int argc, char *argv[]){

  struct timeval startwtime, endwtime;
  double totaltime;

  int k = atoi(argv[1]);
  int n = atoi(argv[2]);

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

  // generate random points
  for(int i = 0; i < n*n; i++){
    G[i] = (rand() % 2) * 2 - 1;
  }

  // timing call of ising start
  gettimeofday(&startwtime, NULL);

  ising(G, w, k, n);

  // timing call of ising end
  gettimeofday(&endwtime, NULL);

  totaltime = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

  FILE *fp = fopen("results.csv","a");
  fprintf(fp, "%d,%d,%f\n", n, k, totaltime);
  fclose(fp);


  return 0;

}
