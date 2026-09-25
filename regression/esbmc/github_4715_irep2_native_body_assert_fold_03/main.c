// generate_ifthenelse gates the `(cond) || (assert(0),0)` fold on the else
// *program* being observationally no-op, not on there being no else -- the
// native arm read the AST instead and missed this shape. The fold now also
// needs the discarded instruction to be a no-op; `g = 1` is not, so both arms
// keep the branch (esbmc/esbmc#7900).
extern int nd(void);
int g;
int main(void)
{
  int c = nd();
  if (c == 2)
  {
    __ESBMC_assert(0, "j");
    g = 1;
  }
  else
  {
  }
  return 0;
}
