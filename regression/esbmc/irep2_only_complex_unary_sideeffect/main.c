#include <assert.h>

int calls = 0;

_Complex double f(void)
{
  calls++;
  return 1.0 + 2.0i;
}

int main(void)
{
  /* The operand of each operator below performs a side effect, so the IREP2
     adjuster declines to lower it and the node stays complex-typed. Do not
     assert over the components of n or c: reading one drags that node into the
     encoder, which has no complex sort and aborts. The property under test is
     the call count, which survives the decline and stays true once the
     operand binding lands and the decline goes away (§90.2). */
  _Complex double n = -f();
  assert(calls == 1);
  _Complex double c = ~f();
  assert(calls == 2);

  return 0;
}
