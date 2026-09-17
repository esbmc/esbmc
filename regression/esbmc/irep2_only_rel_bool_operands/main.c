int nondet_int(void);

int main(void)
{
  int x = nondet_int(), y = nondet_int();

  /* Both operands of the outer == are boolean; the usual arithmetic
     conversions promote them to int before it compares them. */
  __ESBMC_assert((-x == -y) == (x == y), "negation preserves equality");
  return 0;
}
