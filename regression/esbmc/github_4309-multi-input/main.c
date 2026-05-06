#include <assert.h>

// Stress test: 5 nondet inputs of different integer types, with a
// constraint that admits multiple distinct violating tuples.
// Used to confirm the per-witness Inputs line renders mixed types
// correctly and that index ordering is stable across witnesses.

int main(void)
{
  int a;
  unsigned int b;
  char c;
  short d;
  unsigned char e;

  // Constrain to a small reachable subset, otherwise the witness
  // space is enormous and --max-witnesses dominates.
  if (a < 0 || a > 3)
    return 0;
  if (b > 3)
    return 0;
  if (c < 0 || c > 3)
    return 0;
  if (d < 0 || d > 3)
    return 0;
  if (e > 3)
    return 0;

  // Violation when the five values sum to exactly 5.
  // Multiple input tuples satisfy this (e.g. 5+0+0+0+0, 0+5+0+0+0,
  // 1+1+1+1+1, ...).
  assert(a + (int)b + c + d + e != 5);
  return 0;
}
