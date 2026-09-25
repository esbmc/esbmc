// --no-irep2-native-body is the escape hatch that restores the whole-body
// legacy round-trip (esbmc/esbmc#4715). Same body as
// github_4715_irep2_native_body_default_01; the two tests together pin that
// opting out changes no verdict, which is the property the byte-identity gate
// asserts at GOTO level.
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
  assert(classify(0) == 4);
  assert(classify(7) == 2);
  return 0;
}
