#include <assert.h>

int main()
{
  int x = nondet_int();
  assert(x * (-1) == -x);
  return 0;
}
