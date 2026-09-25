// _fail sibling of github_4715_irep2_native_body_dowhile_01 (W1-loc spike
// Phase C, esbmc/esbmc#4715). Pins that a genuine violation reached through a
// natively converted do/while is still reported, not silently dropped: a
// do/while runs its body at least once, so sum_to(0) is 1, not 0.
#include <assert.h>

int sum_to(int n)
{
  int s = 0;
  int i = 0;
  do
  {
    i = i + 1;
    s = s + i;
  } while (i < n);
  return s;
}

int main(void)
{
  assert(sum_to(0) == 0);
  return 0;
}
