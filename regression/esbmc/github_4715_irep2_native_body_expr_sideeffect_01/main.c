// W1-loc spike Phase C (esbmc/esbmc#4715): every side-effecting expression
// statement now converts natively by delegating to the inherited
// remove_sideeffects, rather than only the plain `x = y;` shape. Covers a
// compound assignment, pre/post increment and decrement, and a call statement
// whose result is discarded -- each lowered by a different remove_sideeffects
// sub-case (remove_assignment's synthesized rhs, remove_pre, remove_post,
// do_function_call). f() converts natively end to end.
//
// Native conversion is byte-identical to the fallback by construction, so this
// test cannot detect the handler silently ceasing to fire; it pins that the
// native path computes the right answer. Byte-identity itself is discharged by
// the flag-on/flag-off --goto-functions-only sweep described in the PR.
#include <assert.h>

int g;

void sink(int v)
{
  g = v;
}

int f(int a)
{
  int x = a;
  x += 3;
  x -= 1;
  x *= 2;
  ++x;
  x++;
  --x;
  x--;
  sink(x);
  return x;
}

int main(void)
{
  assert(f(1) == 6);
  assert(g == 6);
  return 0;
}
