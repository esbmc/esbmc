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
void q1(int argc, char* argv[])
{
	if(argc < 2)
		return;

	int N = nondet_int();
  int a[N];

	switch(2)
	{
	case 0: // crescente
	  for(int i=0; i < N; ++i) a[i] = i;
		break;

	case 1: // decrescente
	  for(int i=(N-1); i >= 0; --i) a[i] = N-1-i;
		break;

	case 2: // aleatorio
	  for(int i=0; i < N; ++i) a[i] = i;

    for (int i=0; i<N; i++) {
        int r = i + (nondet_int() % (N-i));
        int temp = a[i];
        a[i] = a[r];
        a[r] = temp;
    }
		break;
	}

//	for(int i=0; i < N; ++i)
//			cout << a[i] << " ";
//	cout << endl << endl;

  bubblesort(N, a);
//  bubblesort1(N, a);

//	for(int i=0; i < N; ++i)
//		cout << a[i] << " ";

}


#endif /* Q1_H_ */
