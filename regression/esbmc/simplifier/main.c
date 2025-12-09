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

  // Symmetric cases (swap the AND operands)
  assert(((a | b) & (a | ~b)) == a);
  assert(((b | a) & (a | ~b)) == a);
  assert(((a | b) & (~b | a)) == a);
  assert(((b | a) & (~b | a)) == a);

  // Alternative with b and ~b swapped
  assert(((a | b) & (a | ~b)) == a);
  assert(((a | b) & (~b | a)) == a);
  assert(((b | a) & (a | ~b)) == a);
  assert(((b | a) & (~b | a)) == a);

  // With ~b on the left
  assert(((a | ~b) & (a | b)) == a);
  assert(((a | ~b) & (b | a)) == a);
  assert(((~b | a) & (a | b)) == a);
  assert(((~b | a) & (b | a)) == a);

  return 0;
}
