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

  _Bool a = nondet_bool();
  _Bool b = nondet_bool();

  // (a | ~b) & (a | b) --> a
  assert(((a | ~b) & (a | b)) == a);

  // (a | ~b) & (a | b) --> a
  assert(((a | ~b) & (a | b)) == a);
  assert(((a | ~b) & (b | a)) == a);
  assert(((~b | a) & (a | b)) == a);
  assert(((~b | a) & (b | a)) == a);

  // (a ^ b) | (a | b) --> a | b
  assert(((a ^ b) | (a | b)) == (a | b));
  assert(((a | b) | (a ^ b)) == (a | b));

  // (a ^ b) | (a | b) --> a | b (all variations)
  assert(((a ^ b) | (a | b)) == (a | b));
  assert(((a ^ b) | (b | a)) == (a | b));
  assert(((b ^ a) | (a | b)) == (a | b));
  assert(((b ^ a) | (b | a)) == (b | a));
  assert(((a | b) | (a ^ b)) == (a | b));
  assert(((b | a) | (a ^ b)) == (b | a));
  assert(((a | b) | (b ^ a)) == (a | b));
  assert(((b | a) | (b ^ a)) == (b | a));

  // ~(a ^ b) | (a | b) --> -1 (all variations)
  assert((~(a ^ b) | (a | b)) == -1);
  assert((~(a ^ b) | (b | a)) == -1);
  assert((~(b ^ a) | (a | b)) == -1);
  assert((~(b ^ a) | (b | a)) == -1);
  assert(((a | b) | ~(a ^ b)) == -1);
  assert(((b | a) | ~(a ^ b)) == -1);
  assert(((a | b) | ~(b ^ a)) == -1);
  assert(((b | a) | ~(b ^ a)) == -1);

  return 0;
}
