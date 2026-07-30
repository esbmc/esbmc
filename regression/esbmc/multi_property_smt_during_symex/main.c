int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  // Violable, and not the last property: under --multi-property with
  // --smt-during-symex this claim is reported as PASSED (#6540).
  __ESBMC_assert(x != 42, "may fail");
  __ESBMC_assert(x >= 0 || x < 0, "holds");
  return 0;
}
