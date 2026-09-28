/* github #7929: the smtlib backend read a negative bitvector value out of the
 * model as an unsigned magnitude, so get() folded (a < 0) to false. */
int nondet_int(void);

int main(void)
{
  int a = nondet_int();
  __ESBMC_assume(a == -4);
  int f = (a < 0);
  __ESBMC_assert(f == 9, "cex");
  return 0;
}
