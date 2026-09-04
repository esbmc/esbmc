/* The failing pair for #7525. `--no-assertions` is swallowed as
 * --smtlib-solver-prog's value, so assertions stay on and this reports FAILED.
 * Honoured, it would report SUCCESSFUL -- the verdict is what pins the
 * swallowing, not just the warning text. */
#include <assert.h>
int main(void)
{
  int x = 0;
  assert(x == 1);
  return 0;
}
