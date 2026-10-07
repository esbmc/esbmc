int main(void)
{
  unsigned initialized;
  unsigned counter = initialized;
  counter++;

  unsigned reset = 0;
  __ESBMC_init_object(&reset);
  unsigned reset_counter = reset;
  reset_counter++;

  __ESBMC_assert(counter == initialized + 1, "local flow");
  __ESBMC_assert(reset_counter == reset + 1, "init object flow");
  return 0;
}
