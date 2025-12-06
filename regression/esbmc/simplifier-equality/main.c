#include <assert.h>

int main()
{
  int x = nondet_int();
  const int c1=1, c2=2;

  assert(((x + c1) == c2) == (x == (c2 - c1)));
  assert((x == (c2 - c1)) == ((x + c1) == c2));

  return 0;
}
