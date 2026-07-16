// _fail sibling of github_4715_irep2_native_body_while_01 (W1-loc spike Phase
// C, esbmc/esbmc#4715). Pins that a genuine violation through a while loop is
// still reported as VERIFICATION FAILED, not silently dropped, under
// --irep2-native-body: sum_to(5) is 10, so the wrong-value assertion is a
// reachable violation.
#include <assert.h>

int sum_to(int n)
{
  int s = 0;
  int i = 0;
  while (i < n)
  {
    s = s + i;
    i = i + 1;
  }
  return s;
}

int main(void)
{
  assert(sum_to(5) == 11);
  return 0;
}
