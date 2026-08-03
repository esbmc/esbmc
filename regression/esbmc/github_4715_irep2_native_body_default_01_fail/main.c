// Negative twin of github_4715_irep2_native_body_default_01: the same natively
// converted body, with an assertion the computation violates. Pins that the
// default native path still reports the violation rather than verifying
// vacuously.
#include <assert.h>

int classify(int n)
{
  int acc = 0;
  for (int i = 0; i < 3; i++)
    acc += i;

  switch (n)
  {
  case 0:
    acc = acc + 1;
    break;
  default:
    acc = acc - 1;
  }

  if (acc > 0)
    return acc;
  else
    return 0;
}

int main(void)
{
  assert(classify(0) == 5);
  return 0;
}
