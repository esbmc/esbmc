#include <cassert>

int nondet_int();
extern void sink(int);

// A call to a bodyless callee is delegated to convert_function_call, which owns
// do_function_call_symbol -- the path that lowers the __ESBMC_* builtins. If
// the delegation dropped or misplaced the ASSUME, x would be unconstrained and
// the assertion below would fail.
int main()
{
  int x = nondet_int();
  __ESBMC_assume(x > 10);
  sink(x);
  assert(x > 5);
  return 0;
}
