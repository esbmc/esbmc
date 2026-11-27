// #9224: an arithmetic shift rounds toward minus infinity, so it differs
// from signed division by 2^k on negative operands.
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  __ESBMC_assert(x / 4 == x >> 2, "x / 4");
  return 0;
}
