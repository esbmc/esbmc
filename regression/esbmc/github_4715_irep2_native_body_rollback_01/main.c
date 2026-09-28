// W1-loc spike Phase C (esbmc/esbmc#4715): pins convert_function's
// tmp_symbol/context rollback (code-reviewer finding on
// feat/w1loc-native-while-sideeffect-cond). first_check()'s side-effecting
// while-condition allocates a return_value$_first_check$N temp mid-attempt
// (do_function_call, via the code_while2t native handler); the trailing
// switch is a deliberately-unsupported statement (no code_switch2t native
// handler) that forces the whole function back to goto_convert_rec. Without
// the rollback, the abandoned attempt's temp allocation would leave
// tmp_symbol.counter advanced, so the legacy fallback's OWN do_function_call
// invocation for the same while-condition would number its temp
// return_value$_first_check$2 instead of $1 — diverging from a flag-off run.
#include <assert.h>

int counter = 0;

int first_check(void)
{
  return counter < 2;
}

int main(void)
{
  int total = 0;
  while (first_check())
  {
    total = total + 1;
    counter = counter + 1;
  }

  // counter == 2 here deterministically (the loop above runs exactly while
  // counter < 2); the switch itself is the deliberately-unsupported
  // statement, not a source of nondeterminism.
  switch (counter)
  {
  case 2:
    total = total + 10;
    break;
  default:
    total = total + 20;
    break;
  }

  assert(total == 12);
  return 0;
}
