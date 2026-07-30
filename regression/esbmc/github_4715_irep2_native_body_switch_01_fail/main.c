// _fail sibling of github_4715_irep2_native_body_switch_01 (W1-loc spike Phase
// C, esbmc/esbmc#4715). Pins that a genuine violation reached through a
// natively converted switch is still reported, not silently dropped: case 1
// falls through into case 2, so classify(1) is 3, not 1.
#include <assert.h>

int classify(int a)
{
  int x = 0;
  switch (a)
  {
  case 1:
    x = 1;
  case 2:
    x = x + 2;
    break;
  default:
    x = 9;
  }
  return x;
}

int main(void)
{
  assert(classify(1) == 1);
  return 0;
}
