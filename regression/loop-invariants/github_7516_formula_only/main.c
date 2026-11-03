/* #7516: a formula-only run decides no base case, so the forward condition
   and inductive step still emit their formulas. */
int main()
{
  int i = 0, s = 0;
  __ESBMC_loop_invariant(i >= 0);
  while (i < 3)
  {
    s++;
    i++;
  }
  __ESBMC_assert(s == 3, "sum");
  return 0;
}
