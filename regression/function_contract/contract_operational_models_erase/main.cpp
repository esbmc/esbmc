#include <vector>
int main() {
  std::vector<int> v;
  unsigned long n = nondet_ulong(); __ESBMC_assume(n >= 3 && n <= 8);
  v.assign(n, 7);
  v[0] = 1; v[1] = 2;
  v.erase(v.begin());
  __ESBMC_assert(v.size() == n - 1 && v[0] == 2 && v[n - 2] == 7, "erased");
#ifdef BAD
  __ESBMC_assert(v[0] == 1, "bad");
#endif
}
