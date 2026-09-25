// _fail sibling of github_4715_irep2_native_body_sideeffect_cond_01 (W1-loc
// spike Phase C, esbmc/esbmc#4715). Pins that a genuine violation reached
// through a natively converted side-effecting loop condition is still reported:
// `while (t--)` runs the body one more time than `while (t)` would, so
// dowhile_post_dec(2) is 3, not 2.
#include <assert.h>

int dowhile_post_dec(int t)
{
  int s = 0;
  do
  {
    s = s + 1;
  } while (t--);
  return s;
}

int main(void)
{
  assert(dowhile_post_dec(2) == 2);
  return 0;
}
