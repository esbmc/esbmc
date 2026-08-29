/* No <assert.h>: `assert` is implicitly declared, so the adjuster synthesises
   the callee symbol. A sideeffect2t carries no location of its own, so the
   symbol's has to come from the enclosing statement. */
int main(void)
{
  int x = 1;

  assert(x == 1);
  return 0;
}
