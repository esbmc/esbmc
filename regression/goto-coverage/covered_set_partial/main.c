int nondet_int(void);

int main(void)
{
  int value = nondet_int();
  __ESBMC_assume(value > 0);
  if (value > 0)
    return 1;
  return 0;
}
