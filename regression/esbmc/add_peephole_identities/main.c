// Peephole identities from #626. Each must hold for every input, including at
// the wrapping boundaries, so the assertions below are the semantics the folds
// have to preserve rather than a check that they fired.
int nondet_int(void);
unsigned nondet_uint(void);

int main(void)
{
  int x = nondet_int();
  int a = nondet_int();
  int b = nondet_int();

  __ESBMC_assert(x + x == (x << 1), "x + x is x << 1");
  __ESBMC_assert(~b + 1 == -b, "~b + 1 is -b");
  __ESBMC_assert((a + 1) + ~b == a - b, "(a + 1) + ~b is a - b");
  __ESBMC_assert(~b + (a + 1) == a - b, "~b + (a + 1) is a - b");
  __ESBMC_assert((a + ~b) + 1 == a - b, "(a + ~b) + 1 is a - b");
  __ESBMC_assert((~b + a) + 1 == a - b, "(~b + a) + 1 is a - b");
  __ESBMC_assert(~b + 3 == 2 - b, "~b + C is (C - 1) - b");

  // The same identities on an unsigned type, where the wrap is the point.
  unsigned u = nondet_uint();
  unsigned v = nondet_uint();
  __ESBMC_assert(u + u == (u << 1), "unsigned x + x is x << 1");
  __ESBMC_assert(~v + 1u == -v, "unsigned ~v + 1 is -v");
  __ESBMC_assert((u + 1u) + ~v == u - v, "unsigned (u + 1) + ~v is u - v");

  return 0;
}
