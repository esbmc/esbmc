#include <string.h>
#include <assert.h>

#define T unsigned long long

int main()
{
  // basic memcpy
  T a0, a1;
  __ESBMC_memcpy(&a0, &a1, sizeof(T));
  assert(a0 == a1);

  return 0;
}
