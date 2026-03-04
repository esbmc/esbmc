#include <assert.h>

#define test_type int
#define type_size sizeof(test_type)
#define array_size 5 // This must be a constant

int main()
{
  test_type arr[array_size];
  __ESBMC_memset(arr, 1, type_size * array_size);
  for(int i = 0; i < array_size; i++)
  {
    unsigned char *p = (unsigned char *)&arr[i];
    for(int j = 0; j < type_size; j++)
      assert(p[j] == 1);
  }
}