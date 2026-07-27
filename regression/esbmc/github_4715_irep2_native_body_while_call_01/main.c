// W1-loc spike Phase C (esbmc/esbmc#4715): pins that a `while` loop whose
// condition is directly a function call (has_sideeffect(cond) true) converts
// natively under --irep2-native-body, delegating to the shared
// generate_conditional_branch/remove_sideeffects helpers instead of falling
// back to goto_convert_rec. The loop body's plain C reassignment statements
// are themselves still unsupported natively (code_expression2t wrapping a
// sideeffect_assign2t), so main()'s conversion attempt allocates a
// return_value$_has_more$N temp for the while-cond call and THEN falls back
// to goto_convert_rec for the whole function — exercising the
// tmp_symbol/context rollback in convert_function (github_4715_irep2_native_
// body_rollback_01 covers the same hazard with an explicit trailing switch).
#include <assert.h>

int counter = 0;

int has_more(void)
{
  return counter < 3;
}

int main(void)
{
  int total = 0;
  while (has_more())
  {
    total = total + 1;
    counter = counter + 1;
  }
  assert(total == 3);
  return 0;
}
