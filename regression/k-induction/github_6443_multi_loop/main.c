/* github.com/esbmc/esbmc/issues/6443
 *
 * Companion to github_6443_single_loop. The assertion lives in the second
 * loop, so the inductive step's `first_loop` iteration counter does not
 * describe it. Converting it to the induction hypothesis here would assume
 * the violation away; the bug must still be found. */
#include <assert.h>
int main(void)
{
  for (int i = 0; i < 10; i++)
    assert(1);

  for (int j = 0; j < 2; j++)
    assert(0);

  return 0;
}
