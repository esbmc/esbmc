#include <assert.h>

int main()
{
  int c = nondet_int();
  int d = nondet_int();
  int e = nondet_int();

  // Pattern: d + c == d + e -> c == e

  // Variation 1: (d + c) == (d + e)
  assert(((d + c) == (d + e)) == (c == e));

  // Variation 2: (d + c) == (e + d) [commutative on right]
  assert(((d + c) == (e + d)) == (c == e));

  // Variation 3: (c + d) == (d + e) [commutative on left]
  assert(((c + d) == (d + e)) == (c == e));

  // Variation 4: (c + d) == (e + d) [both commutative]
  assert(((c + d) == (e + d)) == (c == e));

  int b = nondet_int();

  //  d + b == c + d  ->  b == c
  assert(((d + b) == (c + d)) == (b == c));

  int a = nondet_int();

  // a + c == c + d  ->  a == d
  assert(((a + c) == (c + d)) == (a == d));

  // a + d == c + d  ->  a == c
  assert(((a + d) == (c + d)) == (a == c));

  return 0;
}
