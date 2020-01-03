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

#ifndef ISING_H
#define ISING_H

#include <stdio.h>
#include <stdlib.h>

// auxiliary definitions for row major access
#define new_G(i,j)  *(new_G + (i) * n + (j))
#define G(i,j)      *(G + (i) * n + (j))
#define w(i,j)      *(w + (i) * 5 + (j))
#define test(i,j)   *(test + (i) * n + (j))

//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/
void ising(int *G, double *w, int k, int n);

#endif
