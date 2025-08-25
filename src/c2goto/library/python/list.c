#include <stddef.h>

void ESBMC_py_create_list(unsigned long int *arr, size_t size, unsigned long int value)
{
  size_t i = 0;
  while (i < size)
  {
    arr[i] = value;
  }
}
