// --bitwuzla-opt reaches Bitwuzla's own option parser, which rejects an
// out-of-range value before the solver is built.
int nondet_int(void);

int main()
{
  int x = nondet_int();
  __ESBMC_assert(x != 1234, "x can be anything");
  return 0;
}
