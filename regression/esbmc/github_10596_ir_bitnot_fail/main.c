// The dual of ../github_10596_ir_bitnot: ~x is -x - 1, never -x.
long long nondet_longlong(void);

int main(void)
{
  long long x = nondet_longlong();
  __ESBMC_assert(~x == -x, "~x == -x");
  return 0;
}
