// https://github.com/esbmc/esbmc/issues/9351
#include <assert.h>
#include <stddef.h>
int main(void)
{
  _BitInt(17) a[2];
  ptrdiff_t d = (char *)&a[1] - (char *)&a[0];
  assert(d == 3);
  return 0;
}
