// The IREP2-native goto_convert dispatcher is the default since the W1-loc
// keystone concluded (esbmc/esbmc#4715): this body converts natively with no
// flag on the command line. It mixes the kinds the native ladder covers —
// decl, expression-statement assignment, for, if/else, switch, return — so a
// regression that silently reverted the default would still verify, but the
// companion _optout_ test pins that both paths agree.
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
