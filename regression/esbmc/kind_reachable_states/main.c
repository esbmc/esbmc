// The loop reaches (a, b) = (0,0), (1,0) and (0,1). The interval box also
// allows (1,1), from which the error is 20 steps away, so k-induction needs the
// exact reachable states to prove it (#10013).
#include <assert.h>
extern int __VERIFIER_nondet_int(void);
int a = 0;
int b = 0;
int c = 0;
int step(int in)
{
  if (a == 1 && b == 1)
  {
    c = c + 1;
    if (c >= 20)
      assert(0);
    return 0;
  }
  if (a == 0 && b == 0 && in == 1)
  {
    a = 1;
    return 1;
  }
  if (a == 1 && b == 0 && in == 2)
  {
    a = 0;
    b = 1;
    return 2;
  }
  if (a == 0 && b == 1 && in == 1)
  {
    b = 0;
    return 3;
  }
  return -2;
}
int main(void)
{
  int out = 0;
  while (1)
  {
    int in = __VERIFIER_nondet_int();
    if (in != 1 && in != 2)
      return out;
    out = step(in);
  }
}
