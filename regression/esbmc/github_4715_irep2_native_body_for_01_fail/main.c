// _fail sibling of github_4715_irep2_native_body_for_01 (W1-loc spike Phase C,
// esbmc/esbmc#4715). Pins that a genuine violation reached through a natively
// converted for loop is still reported, not silently dropped: the loop runs
// i = 0..4, so sum_to(5) is 10, not 11.
#include <assert.h>

int sum_to(int n)
{
  int s = 0;
  for (int i = 0; i < n; i = i + 1)
    s = s + i;
  return s;
}

int main(void)
{
  assert(sum_to(5) == 11);
  return 0;
}
