unsigned nondet_uint(void);

int main(void)
{
  unsigned initialized;
  unsigned counter = initialized;
  counter++;

  unsigned reset = 0;
  __ESBMC_init_object(&reset);
  unsigned reset_counter = reset;
  reset_counter++;

  unsigned explicit_nondet = nondet_uint();
  __ESBMC_assert(0, "expected failure");
  return explicit_nondet;
}
