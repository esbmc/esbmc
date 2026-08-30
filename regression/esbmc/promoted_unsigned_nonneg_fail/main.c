/* Soundness twin: a genuinely signed value against zero must NOT fold. */
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  __ESBMC_assert(x >= 0, "a signed nondet can be negative");
  return 0;
}
