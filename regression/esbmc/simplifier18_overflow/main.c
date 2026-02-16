#include <assert.h>

int main()
{
  int c = nondet_int();
  int d = nondet_int();
  int e = nondet_int();

  // (d - c) == (d - e) -> c == e
  assert(((d - c) == (d - e)) == (c == e));

  return 0;
}
