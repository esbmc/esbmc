// --bitwuzla-opt reaches Bitwuzla's own option parser, which rejects an
// out-of-range value before the solver is built.
// KNOWNBUG on the camada branch: camada has no way to pass a raw backend
// option, so --bitwuzla-opt is not registered.
int nondet_int(void);

int main()
{
  int x = nondet_int();
  __ESBMC_assert(x != 1234, "x can be anything");
  return 0;
}
