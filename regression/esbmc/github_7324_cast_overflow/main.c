// The dual of ../github_7324_cast_overflow_fail: a value INT_MAX can hold
// must not be reported. The lower bound needs stating too -- under --ir an
// unsigned symbol is not constrained non-negative on its own.
int main(void)
{
  unsigned int u;
  __ESBMC_assume(u > 10u && u < 2147483648u);
  int i = (int)u;
  return i;
}
