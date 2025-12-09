#include <assert.h>

int main()
{
  int x = nondet_int();
  int y = nondet_int();

  // x + (y - x) -> y
  assert((x + (y - x)) == y);

  // (y - x) + x -> y
  assert(((y - x) + x) == y);  

  // x + -x -> 0
  assert((x + -x) == 0);

  // x + ~x -> -1
  assert((x + ~x) == -1);

  return 0;
}
