// The doubling identity, stated so that proving it needs the multiplication
// itself rather than a constant the frontend already folded. The simplifier
// equivalence workflow plants an unsound `x * 2 -> x` in mul2t::do_simplify()
// and requires this to fail; that is what shows the check still sees per-node
// peepholes (esbmc/esbmc#7260). Here it just pins the fold as sound.
int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x > 0 && x < 100);
  int y = x * 2;
  __ESBMC_assert(y == x + x, "doubling is repeated addition");
  return 0;
}
