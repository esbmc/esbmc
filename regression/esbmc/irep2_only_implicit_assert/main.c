/* No <assert.h>: `assert` is implicitly declared, so the adjuster declares the
   callee itself and must give the symbol the base name goto_convert matches. */
int main(void)
{
  int x = 1;
  assert(x == 1);
  return 0;
}
