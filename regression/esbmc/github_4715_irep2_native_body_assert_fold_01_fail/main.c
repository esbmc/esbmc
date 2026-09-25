// Verdict half of github_4715_irep2_native_body_assert_fold_01, chosen so the
// guard's SIGN is observable: the fold must give the assert the *negated*
// condition, so under `c == 1` it fires. An inverted fold guards it with the
// condition itself, which holds here and would report SUCCESSFUL.
extern int nd(void);
int main(void)
{
  int c = nd();
  __ESBMC_assume(c == 1);
  if (c == 1)
    __ESBMC_assert(0, "reachable");
  return 0;
}
