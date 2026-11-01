#include <vector>
int main() {
  std::vector<int> v;
  unsigned long n = nondet_ulong(); __ESBMC_assume(n >= 1 && n <= 300);
  v.assign(n, 7);
  __ESBMC_assert(v.size() == n && v[n - 1] == 8, "filled");
  return 0;
}
