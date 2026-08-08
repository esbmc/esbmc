// The `(void)((cond) || (assert(0),0))` fold, which is the only assert-fold
// shape the regression corpus actually reaches (github_1565 and three others).
// generate_ifthenelse gates it on the else *program* being observationally
// no-op, not on there being no else -- the native arm read the AST instead and
// missed this shape. The fold discards the branch's second instruction; that is
// legacy behaviour, reproduced deliberately.
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
