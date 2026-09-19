// #1030: vectorised loop reading and writing int arrays through vector
// pointers.
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

typedef int32_t v4si __attribute__((vector_size(16)));

void multiply_arrays(int32_t *arr1, int32_t *arr2, int32_t *result, size_t size)
{
  for (size_t i = 0; i < size; i += 4)
  {
    v4si a = *(v4si *)(arr1 + i);
    v4si b = *(v4si *)(arr2 + i);
    *(v4si *)(result + i) = a * b;
  }
}

int main()
{
  int32_t arr1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t arr2[] = {2, 4, 6, 8, 10, 12, 14, 16};
  int32_t result[8];

  multiply_arrays(arr1, arr2, result, 8);

  for (size_t i = 0; i < 8; i++)
    assert(result[i] == arr1[i] * arr2[i]);
  return 0;
}
