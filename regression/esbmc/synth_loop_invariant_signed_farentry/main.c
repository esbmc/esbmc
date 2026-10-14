/* Entry value far past any admissible bound, so the loop never runs. The
 * accumulator must still read as its entry value at the exit. */
int nondet_int();
int main(void)
{
  int n = nondet_int();
  __ESBMC_assume(n >= 0 && n <= 20);
  int i = 1000, s = 0;
  while (i < n)
  {
    s = s + 1;
    i++;
  }
  __ESBMC_assert(s == 0, "loop never entered");
  return 0;
}
