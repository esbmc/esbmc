// esbmc/esbmc#4715 (V.4.4 parity): the C `assert` macro lowers to a discarded
// statement-level ternary `cond ? 0 : __assert_fail()` whose result type is
// void (empty) while its branches are int and void. Under --irep2-bodies the
// whole body is forward-migrated before goto_convert lowers the ternary, so it
// reaches migrate_expr's if-arm directly; the if2t type invariant
// (result/branch type ids must agree) then aborted GOTO generation for every
// C/C++ program containing a libc assert. The migrate arm now coerces the
// divergent branch to the result type, so the assert lowers cleanly.
#include <assert.h>

int main()
{
  int x = 5;
  assert(x == 5);
  return 0;
}
