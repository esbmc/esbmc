int main()
{
  int x = __VERIFIER_nondet_int();
  __ESBMC_assume(x > 0);
  __ESBMC_assert(x > 0, "x stays positive");
  return 0;
}
