#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main()
{
  int initial_size = 1;
  int *arr = (int *)malloc(initial_size * sizeof(int));

  int size = 0;

  for (int i = 1; i <= 10; i++)
  {
    if (size >= initial_size)
    {
      initial_size += 1;
      arr = (int *)realloc(arr, initial_size * sizeof(int));
    }

    arr[size] = i;
    size++;

    printf("Array contents after adding %d: ", i);
    for (int j = 0; j < size; j++)
    {
      printf("%d ", arr[j]);
    }
    printf("\n");
  }

  for (int j = 0; j < size; j++)
  {
    assert(arr[j] != j + 1);
  }

  free(arr);
  return 0;
}