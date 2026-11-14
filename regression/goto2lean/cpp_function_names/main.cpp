#include <cassert>
unsigned nondet_uint();
static unsigned mid(unsigned lo, unsigned hi) { return lo + (hi - lo) / 2; }
int main() {
  unsigned lo = nondet_uint(), hi = nondet_uint();
  __ESBMC_assume(lo <= hi);
  unsigned m = mid(lo, hi);
  assert(lo <= m && m <= hi);
  return 0;
}
