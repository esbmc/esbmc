/* esbmc/esbmc#7901: calling an undefined __ESBMC-prefixed function is a
 * rejected input, not a crash. __ESBMC_nondet_int is no intrinsic ESBMC has. */
/* Declared, so the reproducer does not rest on an implicit declaration, which
 * C23 removed. */
int __ESBMC_nondet_int(void);

int main(void)
{
  int x = __ESBMC_nondet_int();
  return x;
}
