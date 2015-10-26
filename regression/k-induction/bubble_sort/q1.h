/*
 * q1.h
 *
 *  Created on: Apr 10, 2012
 *      Author: mikhail
 */

#ifndef Q1_H_
#define Q1_H_

#include "bubblesort.h"

int nondet_int();
unsigned int nondet_uint();

void q1(int argc, char* argv[])
{
  if (argc < 2)
    return;

  int N = nondet_int();
  int a[N];

  switch (2)
  {
    case 0: // crescente
      for (int i = 0; i < N; ++i)
        a[i] = i;
      break;

    case 1: // decrescente
      for (int j = (N - 1); j >= 0; --j)
        a[j] = N - 1 - j;
      break;

    case 2: // aleatorio
      for (int k = 0; k < N; ++k)
        a[k] = k;

      for (int l = 0; l < N; l++)
      {
        int r = l + (nondet_uint() % (N - l));
        int temp = a[l];
        a[l] = a[r];
        a[r] = temp;
      }
      break;
  }

  bubblesort(N, a);
}

#endif /* Q1_H_ */
