// Negative counterpart of placement_new_no_init (esbmc/esbmc#6184): after the
// fix, `new (p) T;` reaches a clean counterexample instead of the SIGABRT the
// malformed one-operand comma used to cause.
#include <new>
#include <cassert>

int main()
{
  alignas(int) char buf[sizeof(int)];
  int *p = new (buf) int;
  *p = 42;
  assert(*p == 43); // must fail: the store above wrote 42
  return 0;
}
