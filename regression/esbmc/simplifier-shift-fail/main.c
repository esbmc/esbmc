#include <assert.h>

int main()
{
  int x = nondet_int();
  assert(x << 0 == x);
  return 0;
}
