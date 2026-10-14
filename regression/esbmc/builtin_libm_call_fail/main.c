#include <assert.h>

int main(void)
{
  int e = 0;
  __builtin_frexp(8.0, &e);
  assert(e == 0);
  return 0;
}
