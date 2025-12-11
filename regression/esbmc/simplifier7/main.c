#include <assert.h>

int main()
{
  _Bool a = nondet_bool();
  _Bool b = nondet_bool();

  // (a & ~b) | (a ^ b) --> a ^ b
  assert(((a & ~b) | (a ^ b)) == (a ^ b));
  assert(((a & ~b) | (b ^ a)) == (b ^ a));
  assert(((~b & a) | (a ^ b)) == (a ^ b));
  assert(((~b & a) | (b ^ a)) == (b ^ a));

  // (~a ^ b) | (a & b) --> ~a ^ b
  assert(((~a ^ b) | (a & b)) == (~a ^ b));
  assert(((~a ^ b) | (b & a)) == (~a ^ b));
  assert(((b ^ ~a) | (a & b)) == (b ^ ~a));
  assert(((b ^ ~a) | (b & a)) == (b ^ ~a));

  // (~a | b) | (a ^ b) --> -1
  assert(((~a | b) | (a ^ b)) == -1);
  assert(((~a | b) | (b ^ a)) == -1);
  assert(((b | ~a) | (a ^ b)) == -1);
  assert(((b | ~a) | (b ^ a)) == -1);

  // (~a & b) | ~(a | b) --> ~a
  assert(((~a & b) | ~(a | b)) == ~a);
  assert(((~a & b) | ~(b | a)) == ~a);
  assert(((b & ~a) | ~(a | b)) == ~a);
  assert(((b & ~a) | ~(b | a)) == ~a);

  return 0;
}
