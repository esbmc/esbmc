#include <assert.h>

int main()
{
  int *ptr, offset;
  assert((ptr + offset) - offset == ptr);
  assert(__ESBMC_POINTER_OFFSET(ptr) == __ESBMC_POINTER_OFFSET(ptr));
  assert(__ESBMC_POINTER_OFFSET((ptr + offset) - offset) == __ESBMC_POINTER_OFFSET(ptr));
  assert(__ESBMC_POINTER_OFFSET((ptr + 8) - 8) == __ESBMC_POINTER_OFFSET(ptr));
  assert(__ESBMC_POINTER_OFFSET((ptr + 8/2) - 16/4) == __ESBMC_POINTER_OFFSET(ptr));
  assert(__ESBMC_POINTER_OFFSET((ptr + 8%2) - 16%4) == __ESBMC_POINTER_OFFSET(ptr));
  return 0;
}
