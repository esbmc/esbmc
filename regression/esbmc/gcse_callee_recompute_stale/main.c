#include <assert.h>

unsigned nondet_uint();
unsigned a, b, sink;

void f()
{
  a = a + 1u;
  sink = a + b;
}

// f writes `a` and recomputes `a + b`, which the analysis carries back to the
// caller without assigning the caller's symbol for it (#7992).
int main()
{
  a = nondet_uint();
  b = nondet_uint();
  unsigned x = a + b;
  f();
  unsigned z = a + b;
  assert(z == x + 1u);
}
