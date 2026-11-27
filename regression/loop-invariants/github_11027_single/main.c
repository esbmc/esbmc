/* Regression: GitHub #11027 -- as github_11027, without --multi-property, and
 * entering the loop at idx == 3: the trace shows the value the loop is
 * entered with, not a free havoc value. */
int main()
{
  int x = 0;
  int idx;
  __ESBMC_loop_invariant(x == 0);
  for (idx = 3; idx < 10; ++idx)
  {
    __ESBMC_assert(x == 1, "top: x==1");
  }
}
