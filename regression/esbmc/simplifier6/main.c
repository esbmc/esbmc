#include <assert.h>

int main(void) 
{
  _Bool x = nondet_bool();
  _Bool y = nondet_bool();

  // -------------------------------------------------
  // Boolean algebra laws
  // -------------------------------------------------

  // De Morgan's laws
  assert((~(x & y)) == ((~x) | (~y)));
  assert((~(x | y)) == ((~x) & (~y)));

  // Absorption laws
  assert((x & (x | y)) == x);
  assert(x == (x & (x | y)));
  assert((x | (x & y)) == x);
  assert(x == (x | (x & y)));

  // Double negation
  assert((!!x) == x);

  // Logical De Morgan's laws
  int a = nondet_int();
  int b = nondet_int();

  assert((!(a && b)) == (!a || !b));
  assert((!a || !b) == (!(a && b)));
  assert((!(a || b)) == (!a && !b));

  // -------------------------------------------------
  // Comparison negations
  // -------------------------------------------------
  assert((!(a == b)) == (a != b));
  assert((!(a != b)) == (a == b));
  assert((!(a < b)) == (a >= b));
  assert((!(a <= b)) == (a > b));
  assert((!(a > b)) == (a <= b));
  assert((!(a >= b)) == (a < b));

  // -------------------------------------------------
  // Shift simplifications
  // -------------------------------------------------
  assert((a >> 0) == a);

  // -------------------------------------------------
  // Logical absorption
  // -------------------------------------------------
  assert((a && (a || b)) == (_Bool)a);
  assert((_Bool)a == (a && (a || b)));

  assert((a || (a && b)) == (_Bool)a);
  assert((_Bool)a == (a || (a && b)));

  // -------------------------------------------------
  // Unsigned relational simplification
  // -------------------------------------------------
  unsigned u = nondet_uint();
  assert((u >= 0) == 1);

  return 0;
}

