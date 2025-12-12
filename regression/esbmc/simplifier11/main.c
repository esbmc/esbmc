#include <assert.h>

int main()
{
  _Bool a = nondet_bool();
  _Bool b = nondet_bool();

  // ~(a ^ b) | (a & b) --> ~(a ^ b)
  assert((~(a ^ b) | (a & b)) == (~(a ^ b)));
  assert((~(a ^ b) | (b & a)) == (~(a ^ b)));
  assert(((a & b) | ~(a ^ b)) == (~(a ^ b)));
  assert(((b & a) | ~(a ^ b)) == (~(a ^ b)));

  return 0;
}
