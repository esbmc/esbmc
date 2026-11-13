int nondet_int(void);
int main()
{
  /* x - 1 - INT_MAX folds its constants to the magnitude 2^31, which an int
   * cannot hold: the fold must be refused, not wrapped to x + INT_MIN. */
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x <= 10);
  int y = x - 1 - 2147483647;
  __ESBMC_assert(y >= 0, "y >= 0");
  return 0;
}
