#include <assert.h>
#include <stdlib.h>

#ifndef MAX
# define MAX 7
#endif

void swap(int *xp, int *yp) {
  int temp = *xp;
  *xp = *yp;
  *yp = temp;
}

// A function to implement bubble sort
void bubbleSort(int arr[], int n) {
  int i, j;
  for (i = 0; i < n - 1; i++)
    // Last i elements are already in place
    for (j = 0; j < n - i - 1; j++)
      if (arr[j] > arr[j + 1])
        swap(&arr[j], &arr[j + 1]);
}

int main(void)
{
  int intArray[MAX];

  for (size_t i = 0; i < MAX; i++)
    intArray[i] = nondet_int();

  bubbleSort(intArray, MAX);


  for (size_t i = 0; i < MAX-1; i++)
    assert(intArray[i] <= intArray[i+1]);

  return 0;
}
