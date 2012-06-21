/*
 * bubblesort.h
 *
 *  Created on: Mar 13, 2012
 *      Author: mikhail
 */

#ifndef BUBBLESORT_H_
#define BUBBLESORT_H_

void bubblesort(int size, int item[])
{
	int a, b, t;

	for(a = 1; a < size; ++a)
	{
		for(b = size-1; b >= a; --b)
		{

			if (b-1 < size && b < size)
			{
				if(item[ b - 1] > item[ b ])
				{
					t = item[ b - 1];
					item[ b - 1] = item[ b ];
					item[ b ] = t;
				}
			}
		}
	}
}

void bubblesort1(int size, int item[])
{
	int j, i, pivot;

	for(j = 1; j < size; ++j)
	{
		pivot = item[j];
		i = j - 1;

		while(i >= 0 && item[i] > pivot)
		{
			item[i+1] = item[i];
			i--;
		}
		item[i+1] = pivot;
	}
}

#endif /* BUBBLESORT_H_ */
