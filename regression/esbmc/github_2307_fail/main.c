#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main(void)
{
  int arr1[4096];
  int arr2[4096];

  memcpy(arr1 + 1, arr2 + 2, 4094 * sizeof(int));

  unsigned i = nondet_uint() % 4094;
  assert(arr1[i + 1] == arr2[i + 1]);
  return 0;
}
