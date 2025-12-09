#include <stdlib.h>
int main()
{
  int *arr = malloc(sizeof(int) * 5);
  if(arr == NULL)
    return 0;

  arr[0] = 42;
  arr = realloc(arr, sizeof(int) * 7);
  arr[6] = 42;
}
