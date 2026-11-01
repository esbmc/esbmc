#include <cstdint>

struct Inst
{
  uint32_t a;
  uint32_t b;
};

int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n < 100000);
  Inst *p = new Inst[n];
  p[0].a = 1;
  __ESBMC_assert(p[0].a == 2, "false: stored 1");
  delete[] p;
}
