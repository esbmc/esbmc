#include <assert.h>

int main()
{
  int x = 1;
  int y = 0;
  while(y < 10 && __VERIFIER_nondet_int())
  {
    x = x + y;
    y = y + 1;
  }
  assert(x == y);
  return 0;
}
