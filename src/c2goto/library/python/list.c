#include <stddef.h>

void __ESBMC_py_create_list(int *arr, size_t size, int value)
{
  size_t i = 0;
  while (i < size)
  {
    arr[i] = value;
  }
}
