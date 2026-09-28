#include <cassert>

int hits_a = 0;
int hits_b = 0;

int bump_a()
{
  ++hits_a;
  return 1;
}

int bump_b()
{
  ++hits_b;
  return 2;
}

// A top-level ternary in expression-statement position: convert_expression
// peels it into convert_ifthenelse before remove_sideeffects runs, so the
// dispatcher delegates the statement rather than failing the walk. Exactly one
// arm must be evaluated, and the statements around it stay native.
void pick(int x)
{
  x > 0 ? bump_a() : bump_b();
}

int main()
{
  pick(1);
  assert(hits_a == 1);
  assert(hits_b == 0);

  pick(-1);
  assert(hits_a == 1);
  assert(hits_b == 1);
  return 0;
}
