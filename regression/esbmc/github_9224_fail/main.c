int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  /* Division truncates toward zero; the shift rounds toward minus infinity. */
  __ESBMC_assert(x / 8 == x >> 3, "");
  return 0;
}
