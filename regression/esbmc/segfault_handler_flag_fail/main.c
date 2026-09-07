int main()
{
  int x = __VERIFIER_nondet_int();
  __ESBMC_assume(x > 0);
  __ESBMC_assert(x > 1, "x exceeds one");
  return 0;
}
