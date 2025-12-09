#include <assert.h>
#include <stdlib.h>

int main()
{
  int *p = NULL;
  assert(__ESBMC_r_ok(p, 4)); // invalid dereference
  return 0;
}
